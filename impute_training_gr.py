"""Tum train kuyularinda typewell ve komsu GR imputasyonunu calistirir."""

from pathlib import Path

from typewell_alignment import save_known_tvt_typewell_gr_imputation
from utils import get_all_training_ids, train_folder


def main() -> None:
    output_dir = Path("data/imputed_training_data")
    diagnostics = save_known_tvt_typewell_gr_imputation(
        well_ids=get_all_training_ids(),
        source_dir=train_folder,
        output_dir=output_dir,
    )

    print(f"Saved {len(diagnostics)} wells to {output_dir}")
    print(f"Typewell-imputed rows: {int(diagnostics['imputed_rows'].sum())}")
    print(
        "Neighbor-imputed rows: "
        f"{int(diagnostics['neighbor_imputed_rows'].sum())}"
    )
    print(f"Final missing rows: {int(diagnostics['final_missing_rows'].sum())}")
    print(
        "Invalid typewell calibrations: "
        f"{int((~diagnostics['calibration_is_valid']).sum())}"
    )


if __name__ == "__main__":
    main()
