#include "CUDASmith/StatementMessage.h"

#include <map>
#include <fstream>
#include <ostream>
#include <set>
#include <vector>

#include "Block.h"
#include "CUDASmith/CUDAProgramGenerator.h"
#include "CUDASmith/ExpressionID.h"
#include "CUDASmith/Globals.h"
#include "CUDASmith/MemoryBuffer.h"
#include "Constant.h"
#include "CVQualifiers.h"
#include "Expression.h"
#include "ExpressionFuncall.h"
#include "ExpressionVariable.h"
#include "FunctionInvocation.h"
#include "FunctionInvocationBinary.h"
#include "FunctionInvocationUnary.h"
#include "Lhs.h"
#include "ProbabilityTable.h"
#include "StatementBreak.h"
#include "StatementContinue.h"
#include "StatementIf.h"
#include "Type.h"
#include "Variable.h"
#include "VectorFilter.h"

class CGContext;
class FactMgr;

namespace CUDASmith {
namespace MessagePassing {
namespace {
// Local memory buffer for the messages.
MemoryBuffer *message_buf;
// Holds all the messages used in the program. Will probably never deallocate.
std::vector<Message *> *messages;
// The message type of all the messages.
Type *message_type;
}  // namespace

void Initialise() {
  messages = new std::vector<Message *>();
  // Message type will be created lazily, after all the other initialisation
  // has occured. // TODO
}

void OutputMessageType(std::ostream& out) {
  if (message_type != NULL) OutputStructUnion(message_type, out);
}

Type *CreateRandomMessageType() {
  // Give each type 14 vars, the first four of which are condition flags.
  std::vector<const Type *> types;
  std::vector<CVQualifiers> qfers;
  std::vector<int> bit_len(14, -1);
  const Type *int_type = &Type::get_simple_type(eInt);
  CVQualifiers qfer(std::vector<bool>({false}), std::vector<bool>({false}));
  types.insert(types.end(), 4, int_type);
  qfers.insert(qfers.end(), 4, qfer);
  for (int idx = 0; idx < 10; ++idx) {
    types.push_back(Type::choose_random_simple()->to_unsigned());
    bool is_vol = rnd_flipcoin(50);
    qfers.push_back(CVQualifiers(
        std::vector<bool>({false}), std::vector<bool>({is_vol})));
  }
  return new Type(types, true, false, qfers, bit_len);
}

Message *RandomMessage() {
  // TODO For now just make one Message.
  // Create message type lazily.
  if (!message_type) message_type = CreateRandomMessageType();
  // Create message buffer lazily.
  if (!message_buf)
    message_buf = MemoryBuffer::CreateMemoryBuffer(MemoryBuffer::kLocal,
        "message", message_type, {Constant::make_int(0)}, {1});
  if (messages->empty())
    //    Variable::CreateVariable("message1", message_type, NULL, &qfer)));
    //    CVQualifiers qfer(std::vector<bool>({false}), std::vector<bool>({false}));
    messages->push_back(new Message(message_buf->itemize({0})));
  return (*messages)[0];
}

void CreateMessageOrderings() {
  for (Message *message : *messages) message->CreateOrdering();
  // TODO Multiple messages forming a DAG.
}

void AddMessageVarsToGlobals(Globals *globals) {
  if (!message_buf) return;
  globals->AddLocalMemoryBuffer(message_buf);
  for (Message *message : *messages) {
    globals->AddLocalMemoryBuffer(
        static_cast<MemoryBuffer *>(message->GetMessageVariable()));
  }
}

void OutputMessageEndChecks(std::ostream& out) {
  for (Message *message : *messages) message->OutputEndChecks(out);
}

}  // namespace MessagePassing

void Message::CreateOrdering() {
  // Give each node a unique ID.
  int id = 0;
  for (Node node : nodes_) {
    node->AssignID(id);
    ids_[node] = id++;
  }

  // Start by splitting the nodes into separate lists for each thread. These
  // will be ordered by sequenced-before.
  size_t max_tid = 0;
  for (Node node : nodes_)
    if (node->Gettid() > max_tid) max_tid = node->Gettid();
  sb_graph_.clear();
  sb_graph_.resize(max_tid + 1);
  for (Node node : nodes_)
    sb_graph_[node->Gettid()].push_back(node);

  // Get an iterator for each thread, use these to sequence the nodes.
  std::vector<std::vector<Node>::iterator> iterators;
  for (std::vector<Node>& sb : sb_graph_)
    iterators.push_back(sb.begin());

  // Now iterate until either all or all but one of the iterators have reached
  // the end.
  nodes_order_.clear();
  nodes_async_.clear();
  std::set<size_t> last_tids;
  for (;;) {
    DistributionTable table;
    for (size_t tid = 0; tid < iterators.size(); ++tid)
      if (last_tids.count(tid) == 0)
        table.add_entry(tid, sb_graph_[tid].end() - iterators[tid]);
    if (table.get_max() == 0) break;
    VectorFilter filter(&table);
    size_t next_tid = filter.lookup(rnd_upto(table.get_max()));
    nodes_order_.push_back(*iterators[next_tid]++);
    // Recored and possibly pick an asynchronous update.
    std::set<size_t> used_tids({next_tid});
    if (table.key_to_prob(next_tid) != table.get_max() && rnd_flipcoin(10)) {
      filter.add(next_tid);
      next_tid = filter.lookup(rnd_upto(filter.get_max_prob(), &filter));
      nodes_async_[nodes_order_.back()].insert(*iterators[next_tid]++);
      used_tids.insert(next_tid);
    }
    last_tids = used_tids;
  }

  // The graph is complete now, so we need to make the constraints each node
  // must wait for before it can update.
  using MessagePassing::Constraint;
  using MessagePassing::MakeConstraint;
  // Constraint values for the flags. We have two pairs, as if we have two async
  // nodes, they must signal differenet flags to prevent interfering.
  int flag1 = 0, flag2 = 0, flag3 = 0, flag4 = 0;
  // Map from the nodes to the set of constraints to check for and set.
  std::map<Node, std::vector<
      std::pair<std::set<Constraint>,std::set<Constraint>>>> unlock_map;
  // Variable objects corresponding to the flags in the message.
  Variable *fvar1 = message_var_->field_vars[0];
  Variable *fvar2 = message_var_->field_vars[1];
  Variable *fvar3 = message_var_->field_vars[2];
  Variable *fvar4 = message_var_->field_vars[3];
  // Temp macro for selecting flag set.
  #define SelectConstraint(fp1, fv1, fp2, fv2) \
      !flag_set ? MakeConstraint(fp1, fv1) : MakeConstraint(fp2, fv2)

  std::set<Node> previous;
  int flag_set = 0;
  std::vector<Node>::iterator node_it = nodes_order_.begin();
  for (; node_it != nodes_order_.end(); ++node_it) {
    std::set<Node> async;
    GetAsynchronousNodes(*node_it, &async);
    std::set<Constraint> constraints;
    constraints.insert(SelectConstraint(fvar1, flag1, fvar3, flag3));
    if (previous.size() == 0 || previous.size() == 2)
      constraints.insert(SelectConstraint(fvar2, flag2, fvar4, flag4));
    // If we were using an alternate flag set, or there are two async nodes,
    // then we will signal a different set of flags. The previous value of which
    // must be included in the set of checks.
    bool flip_flag_set = flag_set || async.size() == 2;
    flag_set = flip_flag_set ? (flag_set ? 0 : 1) : flag_set;
    // Each async node must check all the constraints, but signal only one.
    Node node = *node_it;
    std::set<Constraint> checks = constraints;
    if (flip_flag_set)
      checks.insert(SelectConstraint(fvar1, flag1, fvar3, flag3));
    Constraint signal = SelectConstraint(fvar1, ++flag1, fvar3, ++flag3);
    node->MakeWait(unlock_map[node], checks, {signal});
    node->MakeUpdate();
    node->MakeSignal({signal});
    std::vector<Node>::iterator sb_it = GetsbIterator(node) + 1;
    for (; sb_it != sb_graph_[node->Gettid()].end(); ++sb_it)
      unlock_map[*sb_it].push_back(
          std::make_pair(checks, std::set<Constraint>({signal})));
    end_checks_[node->Gettid()].push_back(
        std::make_pair(checks, std::set<Constraint>({signal})));
    if (async.size() == 2) {
      node = *async.rbegin();
      if (node == *node_it) node = *async.begin();
      checks = constraints;
      if (flip_flag_set)
        checks.insert(SelectConstraint(fvar2, flag2, fvar4, flag4));
      signal = SelectConstraint(fvar2, ++flag2, fvar4, ++flag4);
      node->MakeWait(unlock_map[node], checks, {signal});
      node->MakeAsynchronousUpdate(*node_it);
      node->MakeSignal({signal});
      sb_it = GetsbIterator(node) + 1;
      for (; sb_it != sb_graph_[node->Gettid()].end(); ++sb_it)
        unlock_map[*sb_it].push_back(
            std::make_pair(checks, std::set<Constraint>({signal})));
      end_checks_[node->Gettid()].push_back(
          std::make_pair(checks, std::set<Constraint>({signal})));
    }
    previous = async;
  }
  #undef SelectConstraint

  OutputDotty();
}

void Message::OutputDotty() {
  std::ofstream out;
  out.open("CLProg_message.dot");
  out << "digraph G {" << std::endl;
  out << "splines=true;" << std::endl;
  // Put each tids nodes in a vertical line.
  float x_pos = 1.0f;
  for (size_t tid = 0; tid < sb_graph_.size(); ++tid) {
    out << "TStart" << tid << "[shape=plaintext, label=\"T" << tid
        << "\", pos=\"" << x_pos << ",-1.0!\"]" << std::endl;
    float y_pos = 2.0f;
    for (Node node : sb_graph_[tid]) {
      out << "node" << ids_[node]<< "[label=\"" << "node" << ids_[node]
          << "\", pos=\"" << x_pos << ",-" << y_pos << "!\"]" << std::endl;
      y_pos += 1.5f;
    }
    x_pos += 1.5f;
  }
  // Draw sb edge between all the nodes for each tid.
  for (std::vector<Node>& sb : sb_graph_) {
    if (sb.size() < 2) continue;
    std::vector<Node>::iterator prev = sb.begin();
    std::vector<Node>::iterator next = prev + 1;
    for (; next != sb.end(); ++prev, ++next)
      out << "node" << ids_[*prev] << " -> node" << ids_[*next] << std::endl;
  }
  // Draw edges for the ordering.
  std::vector<Node>::iterator prev = nodes_order_.begin();
  std::vector<Node>::iterator next = prev + 1;
  for (; next != nodes_order_.end(); ++prev, ++next)
    out << "node" << ids_[*prev] << " -> node" << ids_[*next]
        << "[color=\"blue\"]" << std::endl;
  // Draw asynchronous edges.
  for (auto& async : nodes_async_)
    for (Node node : async.second)
      out << "node" << ids_[async.first] << " -> node" << ids_[node]
          << "[color=\"orange\",dir=none]" << std::endl;
  // Draw unlock edges. Requires a lot of linear searching.
  prev = nodes_order_.begin();
  next = prev + 1;
  for (; next != nodes_order_.end(); ++prev, ++next) {
    std::set<Node> prev_async;
    GetAsynchronousNodes(*prev, &prev_async);
    std::set<Node> next_async;
    GetAsynchronousNodes(*next, &next_async);
    for (Node prev_node : prev_async) {
      std::vector<Node>::iterator sb_it = GetsbIterator(prev_node) + 1;
      for (; sb_it != sb_graph_[prev_node->Gettid()].end(); ++sb_it) {
        for (Node async : next_async)
          out << "node" << ids_[*sb_it] << " -> node" << ids_[async]
              << "[color=\"red\"]" << std::endl;
      }
    }
  }
  // Finish
  out << '}' << std::endl;
  out.close();
}

namespace {
// Custom statement implementations for outputting in a specific way.
class CrapStatement : public CUDAStatement {
 public:
  explicit CrapStatement(Block *parent) : CUDAStatement(kNone, parent) {}
  void get_blocks(std::vector<const Block *>& blks) const {}
  void get_exprs(std::vector<const Expression *>& exps) const {}
};

// TODO until release/aqcuire/fence is made
class StatementMemFence : public CrapStatement {
 public:
  explicit StatementMemFence(Block *parent) : CrapStatement(parent) {}
  void Output(std::ostream& out, FactMgr *fm, int indent) const { 
    output_tab(out, indent);
    out << "mem_fence(CLK_LOCAL_MEM_FENCE);" << std::endl;
  }
};

// Output if with no false block in a compact way.
class CompactIf : public CrapStatement {
 public:
  CompactIf(Block *parent, Expression *test, Block *if_true)
      : CrapStatement(parent), test_(test), if_true_(if_true) {
  }
  void Output(std::ostream& out, FactMgr *fm, int indent) const {
    output_tab(out, indent);
    out << "if ";
    test_->Output(out);
    out << ' ';
    if (if_true_->stms.size() == 1) {
      if_true_->stms[0]->Output(out, NULL, 0);
      return;
    }
    out << '{' << std::endl;
    for (Statement *st : if_true_->stms)
      st->Output(out, NULL, indent + 1);
    output_tab(out, indent);
    out << '}' << std::endl;
  }
  std::unique_ptr<Expression> test_;
  std::unique_ptr<Block> if_true_;
};

// break/continue with no test expression.
class CompactBreakContinue : public CrapStatement {
 public:
  CompactBreakContinue(Block *parent, bool is_break)
      : CrapStatement(parent), is_break_(is_break) {
  }
  void Output(std::ostream& out, FactMgr *fm, int indent) const {
    output_tab(out, indent);
    out << (is_break_ ? "break" : "continue") << ';' << std::endl;
  }
  bool is_break_;
};
}  // namespace

void Message::OutputEndChecks(std::ostream& out) {
  auto tid_it = end_checks_.begin();
  for (; tid_it != end_checks_.end(); ++tid_it) {
    size_t tid = tid_it->first;
    if (tid_it->second.size() == 0) continue;
    output_tab(out, 1);
    out << "if (";
    ExpressionID(ExpressionID::kLinearLocal).Output(out);
    out << " == " << tid << ") {" << std::endl;
    output_tab(out, 2);
    out << "for (;;) {" << std::endl;
    StatementMemFence(NULL).Output(out, NULL, 3);
    for (const std::pair<std::set<MessagePassing::Constraint>,
                         std::set<MessagePassing::Constraint>>& lock :
        tid_it->second) {
      Block *block = new Block(NULL, 0);
      StatementMessage::MakeConstraintUpdate(lock.second, block);
      CompactIf(NULL, StatementMessage::MakeConstraintCheck(eCmpEq, lock.first),
          block).Output(out, NULL, 3);
    }
    StatementMemFence(NULL).Output(out, NULL, 3);
    // Break out if the constraints for the final unlock and the signal have
    // been exceeded.
    const auto& last_unlock = tid_it->second.back();
    std::set<MessagePassing::Constraint> break_checks = last_unlock.first;
    break_checks.insert(last_unlock.second.begin(), last_unlock.second.end());
    Block dummy(NULL, 0);  // StatementBreak needs this, but doesn't use it :/
    Expression *break_out =
        StatementMessage::MakeConstraintCheck(eCmpGe, break_checks);
    StatementBreak(NULL, *break_out, dummy).Output(out, NULL, 3);
    output_tab(out, 2);
    out << '}' << std::endl;
    output_tab(out, 1);
    out << '}' << std::endl;
  }
}

void Message::GetAsynchronousNodes(Node node, std::set<Node> *async) const {
  assert(node != NULL);
  assert(async != NULL);
  async->clear();
  std::map<Node, std::set<Node>>::const_iterator async_it =
      nodes_async_.find(node);
  if (async_it != nodes_async_.end()) *async = async_it->second;
  async->insert(node);
}

std::vector<StatementMessage *>::iterator Message::GetsbIterator(Node node) {
  std::vector<Node>::iterator it = sb_graph_[node->Gettid()].begin();
  while (*it != node && it != sb_graph_[node->Gettid()].end()) ++it;
  assert(it != sb_graph_[node->Gettid()].end());
  return it;
}

StatementMessage *StatementMessage::make_random(CGContext& cg_context) {
  Message *message = MessagePassing::RandomMessage();
  unsigned int tid_max = CUDAProgramGenerator::get_threads_per_group();
  std::cout << tid_max << std::endl; // TODO deleteme
  // Limit threads to at most 5.
  if (tid_max > 5) tid_max = 5;
  StatementMessage *st_msg = new StatementMessage(
      cg_context.get_current_block(), message, rnd_upto(tid_max));
  message->RegisterUpdate(st_msg);
  return st_msg;
}

void StatementMessage::MakeWait(
    const std::vector<std::pair<std::set<MessagePassing::Constraint>,
                                std::set<MessagePassing::Constraint>>>& unlocks,
    const std::set<MessagePassing::Constraint>& constraints,
    const std::set<MessagePassing::Constraint>& signals) {
  assert(!wait_ && "wait_ already initialised.");
  wait_.reset(new Block(parent, 0));
  // First stage is to synchronise the message.
  // TODO This should use release/acquire, but output full mem_fence for now
  wait_->stms.push_back(new StatementMemFence(wait_.get()));
  // Break out quick if already passed by checking the constraints and signals.
  // There will be some overlap on the constraints, which is acceptable.
  std::set<MessagePassing::Constraint> early_break = constraints;
  early_break.insert(signals.begin(), signals.end());
  wait_->stms.push_back(new StatementBreak(
      wait_.get(), *MakeConstraintCheck(eCmpGe, early_break), *wait_.get()));
  // Check for the conditions of each unlock, and set the unlock constraint.
  // These are not StatementIfs, they should be small and have no false branch.
  for (const std::pair<std::set<MessagePassing::Constraint>,
                       std::set<MessagePassing::Constraint>>& lock : unlocks) {
    Block *block = new Block(wait_.get(), 0);
    MakeConstraintUpdate(lock.second, block);
    wait_->stms.push_back(new CompactIf(
        wait_.get(), MakeConstraintCheck(eCmpEq, lock.first), block));
  }
  // Now check for the constraints to update the message. Third param for
  // continue is the 'loop' block, but no object for the outer loop exists, so
  // set it to wait_.
  Expression *check = new ExpressionFuncall(*new FunctionInvocationUnary(
      eNot, MakeConstraintCheck(eCmpEq, constraints), NULL));
  wait_->stms.push_back(new StatementContinue(
      wait_.get(), *check, *wait_.get()));
}

void StatementMessage::MakeUpdate() {
  assert(!update_ && "update_ already initialised.");
  update_.reset(new Block(parent, 0));
  Variable *message_var = message_->GetMessageVariable();
  unsigned random_bits = rnd_upto(1024);
  size_t idx = 4;
  for (; random_bits; random_bits >>= 1, ++idx) {
    if (!(random_bits & 1)) continue;
    Variable *message_item = message_var->field_vars[idx];
    unsigned rnd_y = rnd_upto(15) + 1;
    update_->stms.push_back(new StatementAssign(update_.get(),
        *new Lhs(*message_item), *Constant::make_int(rnd_y), eAddAssign));
    update_->stms.push_back(new StatementAssign(update_.get(),
        *new Lhs(*message_item), *Constant::make_int(rnd_y), eMulAssign));
  }
}

void StatementMessage::MakeSignal(
    const std::set<MessagePassing::Constraint>& constraints) {
  assert(!signal_ && "signal_ already initialised.");
  signal_.reset(new Block(parent, 0));
  MakeConstraintUpdate(constraints, signal_.get());
  signal_->stms.push_back(new StatementMemFence(signal_.get()));
  signal_->stms.push_back(new CompactBreakContinue(signal_.get(), true));
}

void StatementMessage::MakeAsynchronousUpdate(StatementMessage *node) {
  assert(!update_ && "signal_ already initialised.");
  assert(node != NULL);
  update_.reset(new Block(parent, 0));
  std::set<const Variable *> used_vars;
  node->GetUpdatedVars(&used_vars);
  Variable *message_var = message_->GetMessageVariable();
  for (size_t idx = 4; idx < 14; ++idx) {
    if (used_vars.count(message_var->field_vars[idx]) == 1) continue;
    Variable *message_item = message_var->field_vars[idx];
    unsigned rnd_y = rnd_upto(16);
    update_->stms.push_back(new StatementAssign(update_.get(),
        *new Lhs(*message_item), *Constant::make_int(rnd_y), eAddAssign));
    update_->stms.push_back(new StatementAssign(update_.get(),
        *new Lhs(*message_item), *Constant::make_int(rnd_y), eMulAssign));
  }
}

void StatementMessage::GetUpdatedVars(std::set<const Variable *> *vars) {
  assert(vars != NULL);
  vars->clear();
  if (!update_) return;
  // For now, just assume there are just the two assigns for each var. Would be
  // better to explicitly store the vars used somewhere.
  for (Statement *st : update_->stms) {
    StatementAssign *st_ass = dynamic_cast<StatementAssign *>(st);
    assert(st_ass != NULL);
    vars->insert(st_ass->get_lhs()->get_var());
  }
}

void StatementMessage::get_blocks(std::vector<const Block *>& blks) const {
  if (!wait_ || !update_ || !signal_) return;
  blks.push_back(wait_.get());
  blks.push_back(update_.get());
  blks.push_back(signal_.get());
}

void StatementMessage::Output(std::ostream& out, FactMgr *fm, int indent) const {
  if (!wait_ || !update_ || !signal_) return;
  out << "// NODE START" << std::endl; //TODO deleteme
  // Node is wrapped in a check for local id. Write out manually.
  output_tab(out, indent);
  out << "if (";
  ExpressionID(ExpressionID::kLinearLocal).Output(out);
  out << " == " << tid_ << ") {" << std::endl;
  // Node is also wrapped in an infinite for loop.
  output_tab(out, indent + 1);
  out << "for (;;) { // node" << id_ << std::endl;
  wait_->Output(out, fm, indent + 2);
  update_->Output(out, fm, indent + 2);
  signal_->Output(out, fm, indent + 2);
  output_tab(out, indent + 1);
  out << '}' << std::endl;
  output_tab(out, indent);
  out << '}' << std::endl;
  out << "// NODE END" << std::endl; //TODO deleteme
}

Expression *StatementMessage::MakeConstraintCheck(int binary_op,
    const std::set<MessagePassing::Constraint>& check) {
  eBinaryOps op = static_cast<eBinaryOps>(binary_op);
  std::set<MessagePassing::Constraint>::iterator check_it = check.begin();
  Expression *expr = new ExpressionFuncall(*new FunctionInvocationBinary(op,
      new ExpressionVariable(*check_it->first),
      Constant::make_int(check_it->second), NULL));
  ++check_it;
  for (; check_it != check.end(); ++check_it)
    expr = new ExpressionFuncall(*new FunctionInvocationBinary(eAnd, expr,
        new ExpressionFuncall(*new FunctionInvocationBinary(op,
            new ExpressionVariable(*check_it->first),
            Constant::make_int(check_it->second), NULL)), NULL));
  return expr;
}

void StatementMessage::MakeConstraintUpdate(
    const std::set<MessagePassing::Constraint>& updates, Block *block) {
  for (const MessagePassing::Constraint& update : updates)
    block->stms.push_back(new StatementAssign(
        block, *new Lhs(*update.first), *Constant::make_int(update.second)));
}

}  // namespace CUDASmith
