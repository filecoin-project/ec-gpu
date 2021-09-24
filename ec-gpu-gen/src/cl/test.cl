KERNEL void test_add_32(Scalar32 a, Scalar32 b, GLOBAL Scalar32 *result) {
  *result = Scalar32_add(a, b);
}

KERNEL void test_mul_32(Scalar32 a, Scalar32 b, GLOBAL Scalar32 *result) {
  *result = Scalar32_mul(a, b);
}

KERNEL void test_sub_32(Scalar32 a, Scalar32 b, GLOBAL Scalar32 *result) {
  *result = Scalar32_sub(a, b);
}

KERNEL void test_pow_32(Scalar32 a, uint b, GLOBAL Scalar32 *result) {
  *result = Scalar32_pow(a, b);
}

KERNEL void test_mont_32(Scalar32 a, GLOBAL Scalar32 *result) {
  *result = Scalar32_mont(a);
}

KERNEL void test_unmont_32(Scalar32 a, GLOBAL Scalar32 *result) {
  *result = Scalar32_unmont(a);
}

KERNEL void test_sqr_32(Scalar32 a, GLOBAL Scalar32 *result) {
  *result = Scalar32_sqr(a);
}

KERNEL void test_double_32(Scalar32 a, GLOBAL Scalar32 *result) {
  *result = Scalar32_double(a);
}

////////////
// CUDA doesn't support 64-bit limbs
#ifndef CUDA

KERNEL void test_add_64(Scalar64 a, Scalar64 b, GLOBAL Scalar64 *result) {
  *result = Scalar64_add(a, b);
}

KERNEL void test_mul_64(Scalar64 a, Scalar64 b, GLOBAL Scalar64 *result) {
  *result = Scalar64_mul(a, b);
}

KERNEL void test_sub_64(Scalar64 a, Scalar64 b, GLOBAL Scalar64 *result) {
  *result = Scalar64_sub(a, b);
}

KERNEL void test_pow_64(Scalar64 a, uint b, GLOBAL Scalar64 *result) {
  *result = Scalar64_pow(a, b);
}

KERNEL void test_mont_64(Scalar64 a, GLOBAL Scalar64 *result) {
  *result = Scalar64_mont(a);
}

KERNEL void test_unmont_64(Scalar64 a, GLOBAL Scalar64 *result) {
  *result = Scalar64_unmont(a);
}

KERNEL void test_sqr_64(Scalar64 a, GLOBAL Scalar64 *result) {
  *result = Scalar64_sqr(a);
}

KERNEL void test_double_64(Scalar64 a, GLOBAL Scalar64 *result) {
  *result = Scalar64_double(a);
}
#endif
