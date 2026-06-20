import h5py

def get_h5_structure(file_path):
    """Return a formatted string representing the structure of an HDF5 file."""
    lines = []

    def collect_tree(name, obj):
        indent_level = name.count('/')
        indent = '    ' * indent_level
        if isinstance(obj, h5py.Group):
            lines.append(f"{indent} {name.split('/')[-1] or '/'} [Group]")
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            lines.append(f"{indent} {name.split('/')[-1]} [Dataset] shape={shape}, dtype={dtype}")

    with h5py.File(file_path, 'r') as f:
        lines.append(f"File: {file_path}")
        lines.append("Structure:")
        f.visititems(collect_tree)

    return "\n".join(lines)

def save_structure_to_txt(file_path, output_path):
    """Save the HDF5 structure to a text file."""
    structure_str = get_h5_structure(file_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(structure_str)
    print(f"Structure saved to: {output_path}")

if __name__ == "__main__":
    file_path = "/home/ulrike/OSS-DBSv2/input_files/I_filtered_5000_350_sigma20_Cell_with_AIS.h5"
    output_path = "/home/ulrike/OSS-DBSv2/input_files/I_filtered_5000_350_sigma20_Cell_with_AIS.h5.txt"

    save_structure_to_txt(file_path, output_path)