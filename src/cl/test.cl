__kernel void test_add_32(Fr32 a, Fr32 b, __global Fr32 *result) {
  *result = Fr32_add(a, b);
}

__kernel void test_mul_32(Fr32 a, Fr32 b, __global Fr32 *result) {
  *result = Fr32_mul(a, b);
}

__kernel void test_sub_32(Fr32 a, Fr32 b, __global Fr32 *result) {
  *result = Fr32_sub(a, b);
}

__kernel void test_pow_32(Fr32 a, uint b, __global Fr32 *result) {
  *result = Fr32_pow(a, b);
}

__kernel void test_mont_32(Fr32 a, __global Fr32 *result) {
  *result = Fr32_mont(a);
}

__kernel void test_unmont_32(Fr32 a, __global Fr32 *result) {
  *result = Fr32_unmont(a);
}

__kernel void test_sqr_32(Fr32 a, __global Fr32 *result) {
  *result = Fr32_sqr(a);
}

__kernel void test_double_32(Fr32 a, __global Fr32 *result) {
  *result = Fr32_double(a);
}

////////////

__kernel void test_add_64(Fr64 a, Fr64 b, __global Fr64 *result) {
  *result = Fr64_add(a, b);
}

__kernel void test_mul_64(Fr64 a, Fr64 b, __global Fr64 *result) {
  *result = Fr64_mul(a, b);
}

__kernel void test_sub_64(Fr64 a, Fr64 b, __global Fr64 *result) {
  *result = Fr64_sub(a, b);
}

__kernel void test_pow_64(Fr64 a, uint b, __global Fr64 *result) {
  *result = Fr64_pow(a, b);
}

__kernel void test_mont_64(Fr64 a, __global Fr64 *result) {
  *result = Fr64_mont(a);
}

__kernel void test_unmont_64(Fr64 a, __global Fr64 *result) {
  *result = Fr64_unmont(a);
}

__kernel void test_sqr_64(Fr64 a, __global Fr64 *result) {
  *result = Fr64_sqr(a);
}

__kernel void test_double_64(Fr64 a, __global Fr64 *result) {
  *result = Fr64_double(a);
}
