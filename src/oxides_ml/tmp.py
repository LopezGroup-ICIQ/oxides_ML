def extract_atom_indices(path: str, offline: bool = False) -> list[int]:
    """
    Extract adsorbate atom indices, using cached data or POSCAR parsing.
    Avoids PubChem lookup if offline is True.

    Args:
        path (str): Path to the POSCAR/CONTCAR file.
        offline (bool): If True, skips PubChem requests (useful in multiprocessing).

    Returns:
        list[int]: List of adsorbate atom indices.
    """
    parts = path.split(os.sep)

    if "surface_adsorbates" in parts:
        surface = parts[-6]
        molecule_name = parts[-4]
        context = "surface_adsorbates"
    elif "gas_phase" in parts:
        surface = "gas"
        molecule_name = parts[-2]
        context = "gas_phase"
    else:
        return []

    cache_key = f"{surface}__{molecule_name}__{context}"

    # Use cache if available
    if cache_key in index_cache:
        return index_cache[cache_key]

    # If offline and not cached, we cannot proceed
    if offline:
        raise RuntimeError(f"Adsorbate indices for {cache_key} not in cache and offline=True")

    # Otherwise, fetch formula online
    parsed_formula = get_pubchem_formula(molecule_name)
    if parsed_formula:
        total_atoms = sum(parsed_formula.values())
        atom_indices = get_adsorbate_indices_from_vasp(path, total_atoms)
    else:
        atom_indices = []

    # Cache and return
    index_cache[cache_key] = atom_indices
    with open(CACHE_FILE, "w") as f:
        json.dump(index_cache, f, indent=2)

    return atom_indices
