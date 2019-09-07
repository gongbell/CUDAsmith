// Message Passing
// Need:
//  - Message Handling Class (DAG)
//  - DAG combiner
//  - msg_t
//  - statementMessage
// Gen msgtype -> make_random -> Ordering on each msg -> gen code for each node -> combine
// make_random: -> Select random msg -> pick tid -> register to msg

#ifndef _CUDASMITH_STATEMENTMESSAGE_H_
#define _CUDASMITH_STATEMENTMESSAGE_H_

#include <memory>
#include <ostream>
#include <set>
#include <vector>
#include <utility>

#include "Block.h"
#include "CUDASmith/CUDAStatement.h"

class CGContext;
class Expression;
class FactMgr;
class Type;
class Variable;
namespace CUDASmith {
class Globals;
class Message;
class StatementMessage;
}  // namespace CUDASmith

namespace CUDASmith {

// Top level message passing handling.
// Keeps track of all the messages and combines the DAGs at the end.
// Also handles the message type.
namespace MessagePassing {

// Flag constraint.
// These represent the conditions on which 
typedef std::pair<Variable *, int> Constraint;
inline Constraint MakeConstraint(Variable *flag, int value) {
  return std::make_pair(flag, value);
}

// Initialises the message passing data.
void Initialise();

// Print the type of message_t.
void OutputMessageType(std::ostream& out);

// Create a random type for messages.
Type *CreateRandomMessageType();

// Returns a random Message object, or creates a new one with probability
// inversely proportional to the number already created.
Message *RandomMessage();

// Top level function for going through each message and applying an ordering
// for all the nodes that update the message.
// The ordering will be combined to form a DAG.
void CreateMessageOrderings();

// Adds all the message variables to the struct. For now, it will be a buffer of
// size 1.
void AddMessageVarsToGlobals(Globals *globals);

// Outputs the end constraint checks for each message.
void OutputMessageEndChecks(std::ostream& out);

}  // namespace MessagePassing

// Handles each of the messages that are passed around the program.
// Each StatementMessage in the program that mutates the message managed here
// will be a node in the message passing DAG. At the end of program generation,
// the DAG is created by ordering the nodes.
class Message {
 public:
  explicit Message(Variable *message_var) : message_var_(message_var) {
    // assert var->type == msg_t
  }
  // Moving will invalidate refs.
  Message(Message&& other) = delete;
  Message& operator=(Message&& other) = delete;
  ~Message() {}

  // Each AST node created will be given a message to update and a thread ID
  // that will perform the update. These must be registered to create the
  // ordering.
  void RegisterUpdate(StatementMessage *node) { nodes_.push_back(node); }

  // Run the ordering algorithm on all the nodes that update the message.
  // Instructs each node to generate the code blocks such that they will comply
  // with the chosen ordering.
  void CreateOrdering();

  // Outputs a dotty graph of the ordering in CPLrog_messageN.dot.
  void OutputDotty();

  // Outputs the implicit nodes at the end of each thread that checks an
  // unlocks previous nodes in sequenced-before. Assumed to be in entry.
  // TODO For multiple messages, MP should collect and output them.
  void OutputEndChecks(std::ostream& out);

  Variable *GetMessageVariable() const { return message_var_.get(); }
 private:
  std::unique_ptr<Variable> message_var_;

  // StatementMessage pointers are nodes, which are used everywhere.
  typedef StatementMessage *Node;

  // Helpers for dealing with the graphs.
  // Fills the set with all the nodes that are asynchronous with the passed
  // node. This node must be the key node that is used in the ordering.
  void GetAsynchronousNodes(Node node, std::set<Node> *async) const;
  // Get a sequence-before iterator for the given node at it's position.
  std::vector<Node>::iterator GetsbIterator(Node node);

  // All the registered nodes, unordered.
  std::vector<Node> nodes_;
  // Ids for each node. Helps for identifying nodes in the output.
  std::map<Node, int> ids_;
  // Graph of nodes split into each thread ordered by sequenced-before.
  std::vector<std::vector<Node>> sb_graph_;
  // The ordering
  std::vector<Node> nodes_order_;
  // Asynchronous nodes, they do not appear in nodes_order_.
  // Should probably be a vector.
  std::map<Node, std::set<Node>> nodes_async_;
  // Check to be performed for each thread when it reaches the end.
  std::map<size_t, std::vector<std::pair<
      std::set<MessagePassing::Constraint>,
      std::set<MessagePassing::Constraint>>>> end_checks_;

  DISALLOW_COPY_AND_ASSIGN(Message);
};

// The AST node representing updating the message.
// The code performs the following:
//  - wait: Waits for the flag to indicate that it is allowed to proceed.
//  - update: Changes some of the variables on the message.
//  - signal: Update the message's flag to indicate the message can be received.
class StatementMessage : public CUDAStatement {
 public:
  StatementMessage(Block *parent, Message *message, size_t tid)
      : CUDAStatement(kMessage, parent), message_(message), tid_(tid),
        wait_(), update_(), signal_() {
  }
  // Bad idea to allow moving, Message class uses refs to objects in the DAG.
  StatementMessage(StatementMessage&& other) = delete;
  StatementMessage& operator=(StatementMessage&& other) = delete;
  virtual ~StatementMessage() {}

  // Factory method. Gets a message that will be updated, but does not generate
  // any code until instructed. (Maybe also select tid)
  static StatementMessage *make_random(CGContext& cg_context);

  // Given the set of constraints chosen by Message, we can create the blocks.
  // wait_: If the previous node(s) were skipped, they will be updated to signal
  // the next node in the ordering for the previous node sequenced-before this.
  // Then spin loop until signalled.
  // update_: Update a subset of the variables in the message.
  // signal_: Set the flags to signal the next node in the ordering.
  // The signal in both MakeSignal and the unlock maps should probably only be a
  // single update (not a set).
  //
  // The update section should apply some function, f, to a set of the variables
  // in the message that is dependent on the order in which the updates occur:
  //   f(f(x, y), z) != f(f(x, z), y)
  // Therefore, f must be associative and non-commutative:
  //   f(f(x, y), z) == f(x, f(y, z)) != f(x, f(z, y)) == f(f(x, z), y)
  // Associativity here is important, subtract is not and will give the same
  // result regardless of the order of updates.
  // Choices of f are:
  //   f = \x,y -> (x + y) * y  (or even (x * y) + y)
  //   f = strcat(x, y)  (e.g. x << 4 + y, with 0 <= y < 16)
  // Currently uses the first choice.
  void MakeWait(const std::vector<std::pair<
      std::set<MessagePassing::Constraint>,
      std::set<MessagePassing::Constraint>>>& unlocks,
      const std::set<MessagePassing::Constraint>& constraints,
      const std::set<MessagePassing::Constraint>& signals);
  void MakeUpdate();
  void MakeSignal(const std::set<MessagePassing::Constraint>& constraints);

  // Like MakeUpdate, only the update occurs at the same time as another, so
  // the set of variable in the message must be distinct.
  void MakeAsynchronousUpdate(StatementMessage *node);

  // Gets the set of variables accessed by the update.
  void GetUpdatedVars(std::set<const Variable *> *vars);

  // Assign a unique ID to this node.
  void AssignID(int id) { id_ = id; }

  // Pure virtual in Statement.
  void get_blocks(std::vector<const Block *>& blks) const;
  void get_exprs(std::vector<const Expression *>& exps) const {}

  // Outputs as wait -> update -> signal.
  void Output(std::ostream& out, FactMgr *fm, int indent) const;

  Message *GetMessage() const { return message_; }
  size_t Gettid() const { return tid_; }

  // Helpers for creating the actual statements.
  // Creates an Expression that checks whether all the constraints hold.
  static Expression *MakeConstraintCheck(int binary_op,
      const std::set<MessagePassing::Constraint>& check);
  // Creates an assignment for each constraint and appends the to block.
  static void MakeConstraintUpdate(
      const std::set<MessagePassing::Constraint>& updates, Block *block);

 private:
  Message *message_;
  // The thread that will run the code.
  size_t tid_;
  // ID assigned by message.
  int id_;
  // Each block represents a step in the message handling. These won't be
  // created until the DAG has been created.
  std::unique_ptr<Block> wait_;
  std::unique_ptr<Block> update_;
  std::unique_ptr<Block> signal_;

  DISALLOW_COPY_AND_ASSIGN(StatementMessage);
};

}  // namespace CUDASmith

#endif  // _CUDASMITH_STATEMENTMESSAGE_H_
