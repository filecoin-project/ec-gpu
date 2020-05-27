__kernel void test_add(Fr a, Fr b, __global Fr *result) {
  *result = Fr_add(a, b);
}

__kernel void test_mul(Fr a, Fr b, __global Fr *result) {
  *result = Fr_mul(a, b);
}

__kernel void test_sub(Fr a, Fr b, __global Fr *result) {
  *result = Fr_sub(a, b);
}

__kernel void test_pow(Fr a, uint b, __global Fr *result) {
  *result = Fr_pow(a, b);
}
