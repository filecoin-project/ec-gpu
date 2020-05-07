# ff-cl-gen [![Crates.io](https://img.shields.io/crates/v/ff-cl-gen.svg)](https://crates.io/crates/ff-cl-gen)

OpenCL code generator for finite-field arithmetic over prime fields constructed with Rust [ff](https://github.com/filecoin-project/ff) library.

Notes:
 - Limbs are 64-bit long.
 - The library assumes that the most significant bit of your prime-field is unset. This allows for cheap reductions.

## Usage

Generating OpenCL codes for Bls12-381 Fr elements:

```rust
use paired::bls12_381::Fr;
let src = ff_cl_gen::field::<Fr>("Fr");
```

Generated interface (`FIELD` is substituted with `Fr`):

```c
typedef ulong limb;
#define LIMB_BITS (64)

#define FIELD_LIMBS ... // Number of 64bit limbs for this field
#define FIELD_P ... // Normal form of field modulus
#define FIELD_ONE ... // Montomery form of one
#define FIELD_ZERO ... // Montomery/normal form of zero
#define FIELD_BITS (FIELD_LIMBS * LIMB_BITS)

typedef struct { limb val[FIELD_LIMBS]; } FIELD;

bool FIELD_gte(FIELD a, FIELD b); // Greater than or equal
bool FIELD_eq(FIELD a, FIELD b); // Equal
FIELD FIELD_sub(FIELD a, FIELD b); // Modular subtraction
FIELD FIELD_add(FIELD a, FIELD b); // Modular addition
FIELD FIELD_mul(FIELD a, FIELD b); // Modular multiplication
FIELD FIELD_sqr(FIELD a); // Modular squaring
FIELD FIELD_double(FIELD a); // Modular doubling
FIELD FIELD_pow(FIELD base, uint exponent); // Modular power
FIELD FIELD_pow_lookup(global FIELD *bases, uint exponent); // Modular power with lookup table for bases
FIELD FIELD_mont(FIELD a); // To montgomery form
FIELD FIELD_unmont(FIELD a); // To regular form
bool FIELD_get_bit(FIELD l, uint i); // Get `i`th bit (From most significant digit)
uint FIELD_get_bits(FIELD l, uint skip, uint window); // Get `window` consecutive bits, (Starting from `skip`th bit from most significant digit)
```

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
