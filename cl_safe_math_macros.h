




#define safe_mul_hi_func_int8_t_s_s(_t1,_si1,_si2) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; \
  ((((si1) > ((int8_t)0)) && ((si2) > ((int8_t)0)) && ((si1) > ((INT8_MAX) / (si2)))) || \
  (((si1) > ((int8_t)0)) && ((si2) <= ((int8_t)0)) && ((si2) < ((INT8_MIN) / (si1)))) || \
  (((si1) <= ((int8_t)0)) && ((si2) > ((int8_t)0)) && ((si1) < ((INT8_MIN) / (si2)))) || \
  (((si1) <= ((int8_t)0)) && ((si2) <= ((int8_t)0)) && (si1) != ((int8_t)0)) && ((si2) < ((INT8_MAX) / (si1))))) \
  ? (si1) \
  : mul_hi((si1), (si2));})

#define safe_mad_hi_func_int8_t_s_s_s(_t1,_si1,_si2,_si3) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; _t1 si3 = (_si3) ; \
    ({ _t1 tmp = mul_hi((si1), (si2)) ; \
    ((((tmp) > ((int8_t)0)) && ((si3) > ((int8_t)0)) && ((tmp) > (((int8_t)(INT8_MAX))-(si3)))) || \
    (((tmp) < ((int8_t)0)) && ((si3) < ((int8_t)0)) && ((tmp) < (((int8_t)(INT8_MIN))-(si3))))) \
    ? (si1) \
    : mad_hi((si1), (si2), (si3));}) \
  ;})




#define safe_mul_hi_func_int16_t_s_s(_t1,_si1,_si2) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; \
  ((((si1) > ((int16_t)0)) && ((si2) > ((int16_t)0)) && ((si1) > ((INT16_MAX) / (si2)))) || \
  (((si1) > ((int16_t)0)) && ((si2) <= ((int16_t)0)) && ((si2) < ((INT16_MIN) / (si1)))) || \
  (((si1) <= ((int16_t)0)) && ((si2) > ((int16_t)0)) && ((si1) < ((INT16_MIN) / (si2)))) || \
  (((si1) <= ((int16_t)0)) && ((si2) <= ((int16_t)0)) && (si1) != ((int16_t)0)) && ((si2) < ((INT16_MAX) / (si1))))) \
  ? (si1) \
  : mul_hi((si1), (si2));})

#define safe_mad_hi_func_int16_t_s_s_s(_t1,_si1,_si2,_si3) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; _t1 si3 = (_si3) ; \
    ({ _t1 tmp = mul_hi((si1), (si2)) ; \
    ((((tmp) > ((int16_t)0)) && ((si3) > ((int16_t)0)) && ((tmp) > (((int16_t)(INT16_MAX))-(si3)))) || \
    (((tmp) < ((int16_t)0)) && ((si3) < ((int16_t)0)) && ((tmp) < (((int16_t)(INT16_MIN))-(si3))))) \
    ? (si1) \
    : mad_hi((si1), (si2), (si3));}) \
  ;})




#define safe_mul_hi_func_int32_t_s_s(_t1,_si1,_si2) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; \
  ((((si1) > ((int32_t)0)) && ((si2) > ((int32_t)0)) && ((si1) > ((INT32_MAX) / (si2)))) || \
  (((si1) > ((int32_t)0)) && ((si2) <= ((int32_t)0)) && ((si2) < ((INT32_MIN) / (si1)))) || \
  (((si1) <= ((int32_t)0)) && ((si2) > ((int32_t)0)) && ((si1) < ((INT32_MIN) / (si2)))) || \
  (((si1) <= ((int32_t)0)) && ((si2) <= ((int32_t)0)) && (si1) != ((int32_t)0)) && ((si2) < ((INT32_MAX) / (si1))))) \
  ? (si1) \
  : mul_hi((si1), (si2));})

#define safe_mad_hi_func_int32_t_s_s_s(_t1,_si1,_si2,_si3) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; _t1 si3 = (_si3) ; \
    ({ _t1 tmp = mul_hi((si1), (si2)) ; \
    ((((tmp) > ((int32_t)0)) && ((si3) > ((int32_t)0)) && ((tmp) > (((int32_t)(INT32_MAX))-(si3)))) || \
    (((tmp) < ((int32_t)0)) && ((si3) < ((int32_t)0)) && ((tmp) < (((int32_t)(INT32_MIN))-(si3))))) \
    ? (si1) \
    : mad_hi((si1), (si2), (si3));}) \
  ;})




#define safe_mul_hi_func_int64_t_s_s(_t1,_si1,_si2) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; \
  ((((si1) > ((int64_t)0)) && ((si2) > ((int64_t)0)) && ((si1) > ((INT64_MAX) / (si2)))) || \
  (((si1) > ((int64_t)0)) && ((si2) <= ((int64_t)0)) && ((si2) < ((INT64_MIN) / (si1)))) || \
  (((si1) <= ((int64_t)0)) && ((si2) > ((int64_t)0)) && ((si1) < ((INT64_MIN) / (si2)))) || \
  (((si1) <= ((int64_t)0)) && ((si2) <= ((int64_t)0)) && (si1) != ((int64_t)0)) && ((si2) < ((INT64_MAX) / (si1))))) \
  ? (si1) \
  : mul_hi((si1), (si2));})

#define safe_mad_hi_func_int64_t_s_s_s(_t1,_si1,_si2,_si3) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; _t1 si3 = (_si3) ; \
    ({ _t1 tmp = mul_hi((si1), (si2)) ; \
    ((((tmp) > ((int64_t)0)) && ((si3) > ((int64_t)0)) && ((tmp) > (((int64_t)(INT64_MAX))-(si3)))) || \
    (((tmp) < ((int64_t)0)) && ((si3) < ((int64_t)0)) && ((tmp) < (((int64_t)(INT64_MIN))-(si3))))) \
    ? (si1) \
    : mad_hi((si1), (si2), (si3));}) \
  ;})







#define safe_mul24_func_int32_t_s_s(_t1,_si1,_si2) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; \
  ((((si1) < (-(1<<23))) || ((si1) > ((1<<23)-1)) || ((si2) < (-(1<<23))) || ((si2) > ((1<<23)-1))) || \
  (((si1) > ((int32_t)0)) && ((si2) > ((int32_t)0)) && ((si1) > ((INT32_MAX) / (si2)))) || \
  (((si1) > ((int32_t)0)) && ((si2) <= ((int32_t)0)) && ((si2) < ((INT32_MIN) / (si1)))) || \
  (((si1) <= ((int32_t)0)) && ((si2) > ((int32_t)0)) && ((si1) < ((INT32_MIN) / (si2)))) || \
  (((si1) <= ((int32_t)0)) && ((si2) <= ((int32_t)0)) && ((si1) != ((int32_t)0)) && ((si2) < ((INT32_MAX) / (si1))))) \
  ? (si1) \
  : mul24((si1), (si2));})

#define safe_mad24_func_int32_t_s_s_s(_t1,_si1,_si2,_si3) \
  ({ _t1 si1 = (_si1) ; _t1 si2 = (_si2) ; _t1 si3 = (_si3) ; \
  ((((si1) < (-(1<<23))) || ((si1) > ((1<<23)-1)) || ((si2) < (-(1<<23))) || ((si2) > ((1<<23)-1))) || \
  (((si1) > ((int32_t)0)) && ((si2) > ((int32_t)0)) && ((si1) > ((INT32_MAX) / (si2)))) || \
  (((si1) > ((int32_t)0)) && ((si2) <= ((int32_t)0)) && ((si2) < ((INT32_MIN) / (si1)))) || \
  (((si1) <= ((int32_t)0)) && ((si2) > ((int32_t)0)) && ((si1) < ((INT32_MIN) / (si2)))) || \
  (((si1) <= ((int32_t)0)) && ((si2) <= ((int32_t)0)) && ((si1) != ((int32_t)0)) && ((si2) < ((INT32_MAX) / (si1))))) \
  ? (si1) \
  : ({ _t1 tmp = mul24((si1), (si2)) ; \
    ((((tmp) > ((int32_t)0)) && ((si3) > ((int32_t)0)) && ((tmp) > ((INT32_MAX)-(si3)))) || \
    (((tmp) < ((int32_t)0)) && ((si3) < ((int32_t)0)) && ((tmp) < ((INT32_MIN)-(si3))))) \
    ? (si1) \
    : mad24((si1), (si2), (si3));});})







#define safe_mul24_func_uint32_t_u_u(_t1,_ui1,_ui2) \
  ({ _t1 ui1 = (_ui1) ; _t1 ui2 = (_ui2) ; \
  (((ui1) < (0)) || ((ui1) > (1<<24)) || ((ui2) < (0)) || ((ui2) > (1<<24))) \
  ? (ui1) \
  : mul24((ui1), (ui2));})

#define safe_mad24_func_uint32_t_u_u_u(_t1,_ui1,_ui2,_ui3) \
  ({ _t1 ui1 = (_ui1) ; _t1 ui2 = (_ui2) ; _t1 ui3 = (_ui3) ; \
  (((ui1) < (0)) || ((ui1) > (1<<24)) || ((ui2) < (0)) || ((ui2) > (1<<24))) \
  ? (ui1) \
  : mad24((ui1), (ui2), (ui3));})







#define safe_clamp_func(_t1,_t2,_x,_y,_z) \
  ({ _t1 x = (_x) ; _t2 y = (_y) ; _t2 z = (_z) ; \
  ((y) > (z)) \
  ? (x) \
  : clamp((x), (y), (z));})


