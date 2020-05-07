__kernel void test(__global uint *result) {
  Fr two = Fr_add(Fr_ONE, Fr_ONE);
  Fr eight = Fr_mul(Fr_sqr(two), two);

  Fr a = Fr_pow(two, 123456);
  Fr b = Fr_pow(eight, 41152);
  
  a = Fr_unmont(a);
  a = Fr_mont(a);

  *result = Fr_eq(a, b);
}
