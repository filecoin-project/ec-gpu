use ff::PrimeField;
use itertools::join;

static COMMON_SRC: &str = include_str!("cl/common.cl");
static FIELD_SRC: &str = include_str!("cl/field.cl");

/// Divide anything into 64bit chunks
fn limbs_of<T>(value: T) -> Vec<u64> {
    unsafe {
        std::slice::from_raw_parts(
            &value as *const T as *const u64,
            std::mem::size_of::<T>() / 8,
        )
        .to_vec()
    }
}

/// Calculate the `INV` parameter of Montgomery reduction algorithm for 64bit limbs
/// * `a` - Is the first limb of modulus
fn calc_inv(a: u64) -> u64 {
    let mut inv = 1u64;
    for _ in 0..63 {
        inv = inv.wrapping_mul(inv);
        inv = inv.wrapping_mul(a);
    }
    return inv.wrapping_neg();
}

fn define_field(postfix: &str, limbs: Vec<u64>) -> String {
    format!(
        "#define FIELD_{} ((FIELD){{ {{ {} }} }})",
        postfix,
        join(limbs, ", ")
    )
}

fn params<F>() -> String
where
    F: PrimeField,
{
    let one = limbs_of(F::one()); // Get Montomery form of F::one()
    let p = limbs_of(F::char()); // Get regular form of field modulus
    let limbs = one.len(); // Number of limbs
    let inv = calc_inv(p[0]);
    let limbs_def = format!("#define FIELD_LIMBS {}", limbs);
    let p_def = define_field("P", p);
    let one_def = define_field("ONE", one);
    let zero_def = define_field("ZERO", vec![0u64; limbs]);
    let inv_def = format!("#define FIELD_INV {}", inv);
    let typedef = format!("typedef struct {{ limb val[FIELD_LIMBS]; }} FIELD;");
    join(
        &[limbs_def, one_def, p_def, zero_def, inv_def, typedef],
        "\n",
    )
}

fn field_add_sub<F>() -> String
where
    F: PrimeField,
{
    let mut result = String::new();

    for op in &["sub", "add"] {
        let len = limbs_of(F::one()).len();

        let mut src = format!("FIELD FIELD_{}_(FIELD a, FIELD b) {{\n", op);
        if len > 1 {
            src.push_str("asm(");
            src.push_str(format!("\"{}.cc.u64 %0, %0, %{};\\r\\n\"\n", op, len).as_str());
            for i in 1..len - 1 {
                src.push_str(
                    format!("\"{}c.cc.u64 %{}, %{}, %{};\\r\\n\"\n", op, i, i, len + i).as_str(),
                );
            }
            src.push_str(
                format!(
                    "\"{}c.u64 %{}, %{}, %{};\\r\\n\"\n",
                    op,
                    len - 1,
                    len - 1,
                    2 * len - 1
                )
                .as_str(),
            );
            src.push_str(":");
            let inps = join((0..len).map(|n| format!("\"+l\"(a.val[{}])", n)), ", ");
            src.push_str(inps.as_str());

            src.push_str("\n:");
            let outs = join((0..len).map(|n| format!("\"l\"(b.val[{}])", n)), ", ");
            src.push_str(outs.as_str());
            src.push_str(");\n");
        }
        src.push_str("return a;\n}\n");

        result.push_str(&src);
    }

    result
}

pub fn field<F>(name: &str) -> String
where
    F: PrimeField,
{
    join(
        &[
            COMMON_SRC.to_string(),
            params::<F>(),
            field_add_sub::<F>(),
            String::from(FIELD_SRC),
        ],
        "\n",
    )
    .replace("FIELD", name)
}

#[cfg(test)]
mod tests {}
