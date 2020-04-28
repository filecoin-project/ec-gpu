use ff::PrimeField;
use itertools::join;

static COMMON_SRC: &str = include_str!("cl/common.cl");
static FIELD_SRC: &str = include_str!("cl/field.cl");

/// Divide anything into 64bit chunks
fn limbs_of<T>(value: &T) -> &[u64] {
    unsafe {
        std::slice::from_raw_parts(
            value as *const T as *const u64,
            std::mem::size_of::<T>() / 8,
        )
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

fn params<F>(name: &str) -> String
where
    F: PrimeField,
{
    let one = F::one();
    let one = limbs_of(&one); // Get Montomery from of F::one()
    let p = F::char();
    let p = limbs_of(&p); // Get regular form of field modulus
    let limbs = one.len(); // Number of limbs
    let inv = calc_inv(p[0]);
    let limbs_def = format!("#define {}_LIMBS {}", name, limbs);
    let p_def = format!(
        "#define {}_P (({}){{ {{ {} }} }})",
        name,
        name,
        join(p, ", ")
    );
    let one_def = format!(
        "#define {}_ONE (({}){{ {{ {} }} }})",
        name,
        name,
        join(one, ", ")
    );
    let zero_def = format!(
        "#define {}_ZERO (({}){{ {{ {} }} }})",
        name,
        name,
        join(vec![0u32; limbs], ", ")
    );
    let inv_def = format!("#define {}_INV {}", name, inv);
    let typedef = format!("typedef struct {{ limb val[{}_LIMBS]; }} {};", name, name);
    return format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        limbs_def, one_def, p_def, zero_def, inv_def, typedef
    );
}

fn field_add_sub<F>(name: &str) -> String
where
    F: PrimeField,
{
    let mut result = String::new();
    for op in &["sub", "add"] {
        let one = F::one();
        let len = limbs_of(&one).len();
        let mut src = String::from(format!(
            "{} {}_{}_({} a, {} b) {{\n",
            name, name, op, name, name
        ));

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
    return format!(
        "{}\n{}\n{}\n{}\n",
        COMMON_SRC,
        params::<F>(name),
        field_add_sub::<F>(name),
        String::from(FIELD_SRC).replace("FIELD", name)
    );
}

#[cfg(test)]
mod tests {}
