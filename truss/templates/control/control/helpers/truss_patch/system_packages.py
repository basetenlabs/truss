def system_packages_set(packages: list[str]) -> set[str]:
    pkgs = []
    for sys_pkg_line in packages:
        pkgs.extend(sys_pkg_line.strip().split())
    return set(pkgs)
