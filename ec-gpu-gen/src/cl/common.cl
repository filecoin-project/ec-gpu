// Defines to make the code work with both, CUDA and OpenCL
#ifdef __NVCC__
  #define DEVICE __device__
  #define GLOBAL
  #define KERNEL extern "C" __global__
  #define LOCAL __shared__
  #define CONSTANT __constant__

  #define GET_GLOBAL_ID() blockIdx.x * blockDim.x + threadIdx.x
  #define GET_GROUP_ID() blockIdx.x
  #define GET_LOCAL_ID() threadIdx.x
  #define GET_LOCAL_SIZE() blockDim.x
  #define BARRIER_LOCAL() __syncthreads()

  typedef unsigned char uchar;

  #define CUDA
#else // OpenCL
  #define DEVICE
  #define GLOBAL __global
  #define KERNEL __kernel
  #define LOCAL __local
  #define CONSTANT __constant

  #define GET_GLOBAL_ID() get_global_id(0)
  #define GET_GROUP_ID() get_group_id(0)
  #define GET_LOCAL_ID() get_local_id(0)
  #define GET_LOCAL_SIZE() get_local_size(0)
  #define BARRIER_LOCAL() barrier(CLK_LOCAL_MEM_FENCE)
#endif

#ifdef __NV_CL_C_VERSION
#define OPENCL_NVIDIA
#endif

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__)
#define AMD
#endif

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    ulong lo, hi;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64 %1, %2, %3, 0;\r\n"
        "add.cc.u64 %0, %0, %5;\r\n"
        "addc.u64 %1, %1, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
    *d = hi;
    return lo;
  #else
    ulong lo = a * b + c;
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
  #endif
}

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    ulong lo, hi;
    asm("add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64 %1, 0, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));
    *b = hi;
    return lo;
  #else
    ulong lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d) {
  ulong res = (ulong)a * b + c + *d;
  *d = res >> 32;
  return res;
}

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    uint lo, hi;
    asm("add.cc.u32 %0, %2, %3;\r\n"
        "addc.u32 %1, 0, 0;\r\n"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(*b));
    *b = hi;
    return lo;
  #else
    uint lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Reverse the given bits. It's used by the FFT kernel.
DEVICE uint bitreverse(uint n, uint bits) {
  uint r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

#ifdef CUDA
// CUDA doesn't support local buffers ("dynamic shared memory" in CUDA lingo) as function
// arguments, but only a single globally defined extern value. Use `uchar` so that it is always
// allocated by the number of bytes.
extern LOCAL uchar cuda_shared[];

typedef uint uint32_t;
typedef int  int32_t;
typedef uint limb;

DEVICE inline uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}


DEVICE inline uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

typedef struct {
  int32_t _position;
} chain_t;

DEVICE inline
void chain_init(chain_t *c) {
  c->_position = 0;
}

DEVICE inline
uint32_t chain_add(chain_t *ch, uint32_t a, uint32_t b) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=add_cc(a, b);
  else
    r=addc_cc(a, b);
  return r;
}

DEVICE inline
uint32_t chain_madlo(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madlo_cc(a, b, c);
  else
    r=madloc_cc(a, b, c);
  return r;
}

DEVICE inline
uint32_t chain_madhi(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madhi_cc(a, b, c);
  else
    r=madhic_cc(a, b, c);
  return r;
}
#endif
