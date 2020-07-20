use crate::Limb;
use ff::PrimeField;
use itertools::*;

/// Generates PTX-Assembly implementation of FIELD_add_/FIELD_sub_
pub fn field_add_sub_32_amd<F, L: Limb>() -> String
where
    F: PrimeField,
{
    let mut result = String::new();

    if L::bits() == 32 {
    result.push_str("#ifdef AMD\n");
    for op in &["SUB", "ADD"] {
        let len = L::limbs_of(F::one()).len();

        let mut src = format!("FIELD FIELD_{}_amd(FIELD a, FIELD b) {{\n", op);
        if len > 1 {
            src.push_str("__asm volatile(");
            src.push_str(format!("\"S_{}_U32 %0, %0, %{};\\n\"\n", op, len).as_str());
            for i in 1..len {
                src.push_str(
                    format!(
                        "\"S_{}_U32 %{}, %{}, %{};\\n\"\n",
                        if *op == "ADD" {"ADDC"} else {"SUBB"},
                        i,
                        i,
                        len + i
                    )
                    .as_str(),
                );
            }
            src.push_str(":");
            let inps = join(
                (0..len).map(|n| format!("\"+v\"(a.val[{}])", n)),
                ", ",
            );
            src.push_str(inps.as_str());

            src.push_str("\n:");
            let outs = join(
                (0..len).map(|n| format!("\"v\"(b.val[{}])", n)),
                ", ",
            );
            src.push_str(outs.as_str());
            src.push_str(");\n");
        }
        src.push_str("return a;\n}\n");

        result.push_str(&src);
    }
    result.push_str("#endif\n");
    }

    result
}
