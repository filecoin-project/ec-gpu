DEVICE uint bitreverse(uint n, uint bits) {
  uint r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
KERNEL void radix_fft(GLOBAL FIELD* x, // Source buffer
                      GLOBAL FIELD* y, // Destination buffer
                      GLOBAL FIELD* pq, // Precalculated twiddle factors
                      GLOBAL FIELD* omegas, // [omega, omega^2, omega^4, ...]
                      LOCAL FIELD* u_arg, // Local buffer to store intermediary values
                      uint n, // Number of elements
                      uint lgp, // Log2 of `p` (Read more in the link above)
                      uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                      uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{
// CUDA doesn't support local buffers ("shared memory" in CUDA lingo) as function arguments,
// ignore that argument and declare the external memory here instead.
#ifdef CUDA
  extern LOCAL FIELD u[];
#else
   LOCAL FIELD* u = u_arg;
#endif

  uint lid = GET_LOCAL_ID();
  uint lsize = GET_LOCAL_SIZE();
  uint index = GET_GROUP_ID();
  uint t = n >> deg;
  uint p = 1 << lgp;
  uint k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint count = 1 << deg; // 2^deg
  uint counth = count >> 1; // Half of count

  uint counts = count / lsize * lid;
  uint counte = counts + count / lsize;

  // Compute powers of twiddle
  const FIELD twiddle = FIELD_pow_lookup(omegas, (n >> lgp >> deg) * k);
  FIELD tmp = FIELD_pow(twiddle, counts);
  for(uint i = counts; i < counte; i++) {
    u[i] = FIELD_mul(tmp, x[i*t]);
    tmp = FIELD_mul(tmp, twiddle);
  }
  BARRIER_LOCAL();

  const uint pqshift = max_deg - deg;
  for(uint rnd = 0; rnd < deg; rnd++) {
    const uint bit = counth >> rnd;
    for(uint i = counts >> 1; i < counte >> 1; i++) {
      const uint di = i & (bit - 1);
      const uint i0 = (i << 1) - di;
      const uint i1 = i0 + bit;
      tmp = u[i0];
      u[i0] = FIELD_add(u[i0], u[i1]);
      u[i1] = FIELD_sub(tmp, u[i1]);
      if(di != 0) u[i1] = FIELD_mul(pq[di << rnd << pqshift], u[i1]);
    }

    BARRIER_LOCAL();
  }

  for(uint i = counts >> 1; i < counte >> 1; i++) {
    y[i*p] = u[bitreverse(i, deg)];
    y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
  }
}

/// Multiplies all of the elements by `field`
KERNEL void mul_by_field(GLOBAL FIELD* elements,
                        uint n,
                        FIELD field) {
  const uint gid = GET_GLOBAL_ID();
  elements[gid] = FIELD_mul(elements[gid], field);
}
