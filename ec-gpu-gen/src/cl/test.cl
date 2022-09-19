KERNEL void test_add(FIELD a, FIELD b, GLOBAL FIELD *result) {
  *result = FIELD_add(a, b);
}

KERNEL void test_mul(FIELD a, FIELD b, GLOBAL FIELD *result) {
  *result = FIELD_mul(a, b);
}

KERNEL void test_sub(FIELD a, FIELD b, GLOBAL FIELD *result) {
  *result = FIELD_sub(a, b);
}

KERNEL void test_pow(FIELD a, uint b, GLOBAL FIELD *result) {
  *result = FIELD_pow(a, b);
}

KERNEL void test_mont(FIELD a, GLOBAL FIELD *result) {
  *result = FIELD_mont(a);
}

KERNEL void test_unmont(FIELD a, GLOBAL FIELD *result) {
  *result = FIELD_unmont(a);
}

KERNEL void test_sqr(FIELD a, GLOBAL FIELD *result) {
  *result = FIELD_sqr(a);
}

KERNEL void test_double(FIELD a, GLOBAL FIELD *result) {
  *result = FIELD_double(a);
}
