__kernel void test_add_32(Scalar32 a, Scalar32 b, __global Scalar32 *result) {
  *result = Scalar32_add(a, b);
}

__kernel void test_mul_32(Scalar32 a, Scalar32 b, __global Scalar32 *result) {
  *result = Scalar32_mul(a, b);
}

__kernel void test_sub_32(Scalar32 a, Scalar32 b, __global Scalar32 *result) {
  *result = Scalar32_sub(a, b);
}

__kernel void test_pow_32(Scalar32 a, uint b, __global Scalar32 *result) {
  *result = Scalar32_pow(a, b);
}

__kernel void test_mont_32(Scalar32 a, __global Scalar32 *result) {
  *result = Scalar32_mont(a);
}

__kernel void test_unmont_32(Scalar32 a, __global Scalar32 *result) {
  *result = Scalar32_unmont(a);
}

__kernel void test_sqr_32(Scalar32 a, __global Scalar32 *result) {
  *result = Scalar32_sqr(a);
}

__kernel void test_double_32(Scalar32 a, __global Scalar32 *result) {
  *result = Scalar32_double(a);
}

////////////

__kernel void test_add_64(Scalar64 a, Scalar64 b, __global Scalar64 *result) {
  *result = Scalar64_add(a, b);
}

__kernel void test_mul_64(Scalar64 a, Scalar64 b, __global Scalar64 *result) {
  *result = Scalar64_mul(a, b);
}

__kernel void test_sub_64(Scalar64 a, Scalar64 b, __global Scalar64 *result) {
  *result = Scalar64_sub(a, b);
}

__kernel void test_pow_64(Scalar64 a, uint b, __global Scalar64 *result) {
  *result = Scalar64_pow(a, b);
}

__kernel void test_mont_64(Scalar64 a, __global Scalar64 *result) {
  *result = Scalar64_mont(a);
}

__kernel void test_unmont_64(Scalar64 a, __global Scalar64 *result) {
  *result = Scalar64_unmont(a);
}

__kernel void test_sqr_64(Scalar64 a, __global Scalar64 *result) {
  *result = Scalar64_sqr(a);
}

__kernel void test_double_64(Scalar64 a, __global Scalar64 *result) {
  *result = Scalar64_double(a);
}
