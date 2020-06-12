// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define FIELD_BITS (FIELD_LIMBS * LIMB_BITS)

// Greater than or equal
bool FIELD_gte(FIELD a, FIELD b) {
  for(char i = FIELD_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
bool FIELD_eq(FIELD a, FIELD b) {
  for(uchar i = 0; i < FIELD_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#ifdef NVIDIA
  #define FIELD_add_ FIELD_add_nvidia
  #define FIELD_sub_ FIELD_sub_nvidia
#else
  FIELD FIELD_add_(FIELD a, FIELD b) {
    bool carry = 0;
    for(uchar i = 0; i < FIELD_LIMBS; i++) {
      limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  FIELD FIELD_sub_(FIELD a, FIELD b) {
    bool borrow = 0;
    for(uchar i = 0; i < FIELD_LIMBS; i++) {
      limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
FIELD FIELD_sub(FIELD a, FIELD b) {
  FIELD res = FIELD_sub_(a, b);
  if(!FIELD_gte(a, b)) res = FIELD_add_(res, FIELD_P);
  return res;
}

// Modular addition
FIELD FIELD_add(FIELD a, FIELD b) {
  FIELD res = FIELD_add_(a, b);
  if(FIELD_gte(res, FIELD_P)) res = FIELD_sub_(res, FIELD_P);
  return res;
}

// Modular multiplication
FIELD FIELD_mul(FIELD a, FIELD b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  limb t[FIELD_LIMBS + 2] = {0};
  for(uchar i = 0; i < FIELD_LIMBS; i++) {
    limb carry = 0;
    for(uchar j = 0; j < FIELD_LIMBS; j++)
      t[j] = mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[FIELD_LIMBS] = add_with_carry(t[FIELD_LIMBS], &carry);
    t[FIELD_LIMBS + 1] = carry;

    carry = 0;
    limb m = FIELD_INV * t[0];
    mac_with_carry(m, FIELD_P.val[0], t[0], &carry);
    for(uchar j = 1; j < FIELD_LIMBS; j++)
      t[j - 1] = mac_with_carry(m, FIELD_P.val[j], t[j], &carry);

    t[FIELD_LIMBS - 1] = add_with_carry(t[FIELD_LIMBS], &carry);
    t[FIELD_LIMBS] = t[FIELD_LIMBS + 1] + carry;
  }

  FIELD result;
  for(uchar i = 0; i < FIELD_LIMBS; i++) result.val[i] = t[i];

  if(FIELD_gte(result, FIELD_P)) result = FIELD_sub_(result, FIELD_P);

  return result;
}

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
FIELD FIELD_sqr(FIELD a) {
  return FIELD_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of FIELD_add(a, a)
FIELD FIELD_double(FIELD a) {
  for(uchar i = FIELD_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(FIELD_gte(a, FIELD_P)) a = FIELD_sub_(a, FIELD_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
FIELD FIELD_pow(FIELD base, uint exponent) {
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
FIELD FIELD_pow_lookup(__global FIELD *bases, uint exponent) {
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

FIELD FIELD_mont(FIELD a) {
  return FIELD_mul(a, FIELD_R2);
}

FIELD FIELD_unmont(FIELD a) {
  FIELD one = FIELD_ZERO;
  one.val[0] = 1;
  return FIELD_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
bool FIELD_get_bit(FIELD l, uint i) {
  return (l.val[FIELD_LIMBS - 1 - i / LIMB_BITS] >> (LIMB_BITS - 1 - (i % LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
uint FIELD_get_bits(FIELD l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= FIELD_get_bit(l, skip + i);
  }
  return ret;
}

void FIELD_print(FIELD a) {
  printf("0x");
  for (uint i = 0; i < FIELD_LIMBS; i++) {
    printf("%016lx", a.val[FIELD_LIMBS - i - 1]);
  }
}
