// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define FIELD_BITS (FIELD_LIMBS * FIELD_LIMB_BITS)
#if FIELD_LIMB_BITS == 32
  #define FIELD_mac_with_carry mac_with_carry_32
  #define FIELD_add_with_carry add_with_carry_32
#elif FIELD_LIMB_BITS == 64
  #define FIELD_mac_with_carry mac_with_carry_64
  #define FIELD_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool FIELD_gte(FIELD a, FIELD b) {
  for(char i = FIELD_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool FIELD_eq(FIELD a, FIELD b) {
  for(uchar i = 0; i < FIELD_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(OPENCL_NVIDIA) || defined(CUDA)
  #define FIELD_add_ FIELD_add_nvidia
  #define FIELD_sub_ FIELD_sub_nvidia
#else
  DEVICE FIELD FIELD_add_(FIELD a, FIELD b) {
    bool carry = 0;
    for(uchar i = 0; i < FIELD_LIMBS; i++) {
      FIELD_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  FIELD FIELD_sub_(FIELD a, FIELD b) {
    bool borrow = 0;
    for(uchar i = 0; i < FIELD_LIMBS; i++) {
      FIELD_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE FIELD FIELD_sub(FIELD a, FIELD b) {
  FIELD res = FIELD_sub_(a, b);
  if(!FIELD_gte(a, b)) res = FIELD_add_(res, FIELD_P);
  return res;
}

// Modular addition
DEVICE FIELD FIELD_add(FIELD a, FIELD b) {
  FIELD res = FIELD_add_(a, b);
  if(FIELD_gte(res, FIELD_P)) res = FIELD_sub_(res, FIELD_P);
  return res;
}


#ifdef CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void FIELD_reduce(uint32_t accLow[FIELD_LIMBS], uint32_t np0, uint32_t fq[FIELD_LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = FIELD_LIMBS;
  uint32_t accHigh[FIELD_LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void FIELD_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = FIELD_LIMBS;
  const uint32_t yLimbs  = FIELD_LIMBS;
  const uint32_t xyLimbs = FIELD_LIMBS * 2;
  uint32_t temp[FIELD_LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE FIELD FIELD_mul_nvidia(FIELD a, FIELD b) {
  // Perform full multiply
  limb ab[2 * FIELD_LIMBS];
  FIELD_mult_v1(a.val, b.val, ab);

  uint32_t io[FIELD_LIMBS];
  #pragma unroll
  for(int i=0;i<FIELD_LIMBS;i++) {
    io[i]=ab[i];
  }
  FIELD_reduce(io, FIELD_INV, FIELD_P.val);

  // Add io to the upper words of ab
  ab[FIELD_LIMBS] = add_cc(ab[FIELD_LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < FIELD_LIMBS - 1; j++) {
    ab[j + FIELD_LIMBS] = addc_cc(ab[j + FIELD_LIMBS], io[j]);
  }
  ab[2 * FIELD_LIMBS - 1] = addc(ab[2 * FIELD_LIMBS - 1], io[FIELD_LIMBS - 1]);

  FIELD r;
  #pragma unroll
  for (int i = 0; i < FIELD_LIMBS; i++) {
    r.val[i] = ab[i + FIELD_LIMBS];
  }

  if (FIELD_gte(r, FIELD_P)) {
    r = FIELD_sub_(r, FIELD_P);
  }

  return r;
}

#endif

// Modular multiplication
DEVICE FIELD FIELD_mul_default(FIELD a, FIELD b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  FIELD_limb t[FIELD_LIMBS + 2] = {0};
  for(uchar i = 0; i < FIELD_LIMBS; i++) {
    FIELD_limb carry = 0;
    for(uchar j = 0; j < FIELD_LIMBS; j++)
      t[j] = FIELD_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[FIELD_LIMBS] = FIELD_add_with_carry(t[FIELD_LIMBS], &carry);
    t[FIELD_LIMBS + 1] = carry;

    carry = 0;
    FIELD_limb m = FIELD_INV * t[0];
    FIELD_mac_with_carry(m, FIELD_P.val[0], t[0], &carry);
    for(uchar j = 1; j < FIELD_LIMBS; j++)
      t[j - 1] = FIELD_mac_with_carry(m, FIELD_P.val[j], t[j], &carry);

    t[FIELD_LIMBS - 1] = FIELD_add_with_carry(t[FIELD_LIMBS], &carry);
    t[FIELD_LIMBS] = t[FIELD_LIMBS + 1] + carry;
  }

  FIELD result;
  for(uchar i = 0; i < FIELD_LIMBS; i++) result.val[i] = t[i];

  if(FIELD_gte(result, FIELD_P)) result = FIELD_sub_(result, FIELD_P);

  return result;
}

#ifdef CUDA
DEVICE FIELD FIELD_mul(FIELD a, FIELD b) {
  return FIELD_mul_nvidia(a, b);
}
#else
DEVICE FIELD FIELD_mul(FIELD a, FIELD b) {
  return FIELD_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE FIELD FIELD_sqr(FIELD a) {
  return FIELD_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of FIELD_add(a, a)
DEVICE FIELD FIELD_double(FIELD a) {
  for(uchar i = FIELD_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (FIELD_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(FIELD_gte(a, FIELD_P)) a = FIELD_sub_(a, FIELD_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE FIELD FIELD_pow(FIELD base, uint exponent) {
  FIELD res = FIELD_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = FIELD_mul(res, base);
    exponent = exponent >> 1;
    base = FIELD_sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE FIELD FIELD_pow_lookup(GLOBAL FIELD *bases, uint exponent) {
  FIELD res = FIELD_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = FIELD_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE FIELD FIELD_mont(FIELD a) {
  return FIELD_mul(a, FIELD_R2);
}

DEVICE FIELD FIELD_unmont(FIELD a) {
  FIELD one = FIELD_ZERO;
  one.val[0] = 1;
  return FIELD_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool FIELD_get_bit(FIELD l, uint i) {
  return (l.val[FIELD_LIMBS - 1 - i / FIELD_LIMB_BITS] >> (FIELD_LIMB_BITS - 1 - (i % FIELD_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint FIELD_get_bits(FIELD l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= FIELD_get_bit(l, skip + i);
  }
  return ret;
}
