use crate::Limb;
use ff::PrimeField;
use itertools::*;

/// Generates PTX-Assembly implementation of FIELD_add_/FIELD_sub_
pub fn field_add_sub_nvidia<F, L: Limb>() -> String
where
    F: PrimeField,
{
    let mut result = String::new();
    let (ptx_type, ptx_reg) = L::ptx_info();

    result.push_str("#ifdef NVIDIA\n");
    for op in &["sub", "add"] {
        let len = L::limbs_of(F::one()).len();

        let mut src = format!("FIELD FIELD_{}_nvidia(FIELD a, FIELD b) {{\n", op);
        if len > 1 {
            src.push_str("asm(");
            src.push_str(format!("\"{}.cc.{} %0, %0, %{};\\r\\n\"\n", op, ptx_type, len).as_str());
            for i in 1..len - 1 {
                src.push_str(
                    format!(
                        "\"{}c.cc.{} %{}, %{}, %{};\\r\\n\"\n",
                        op,
                        ptx_type,
                        i,
                        i,
                        len + i
                    )
                    .as_str(),
                );
            }
            src.push_str(
                format!(
                    "\"{}c.{} %{}, %{}, %{};\\r\\n\"\n",
                    op,
                    ptx_type,
                    len - 1,
                    len - 1,
                    2 * len - 1
                )
                .as_str(),
            );
            src.push_str(":");
            let inps = join(
                (0..len).map(|n| format!("\"+{}\"(a.val[{}])", ptx_reg, n)),
                ", ",
            );
            src.push_str(inps.as_str());

            src.push_str("\n:");
            let outs = join(
                (0..len).map(|n| format!("\"{}\"(b.val[{}])", ptx_reg, n)),
                ", ",
            );
            src.push_str(outs.as_str());
            src.push_str(");\n");
        }
        src.push_str("return a;\n}\n");

        result.push_str(&src);
    }
    result.push_str("#endif\n");

    result
}
