__kernel void test(__global uint *result) {
  Fr two = Fr_add(Fr_ONE, Fr_ONE);
  Fr eight = Fr_mul(Fr_sqr(two), two);

  Fr a = Fr_pow(two, 123456);
  Fr b = Fr_pow(eight, 41152);

  *result = Fr_eq(a, b);
}
