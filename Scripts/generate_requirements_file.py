import pkg_resources
import types

# Define the function to get imports
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]
            
        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]
            
        yield name

# Define the function to update requirements file
def update_requirements_file(file_path):
    # Read existing requirements from file
    existing_requirements = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                existing_requirements.add(line.strip())
    except FileNotFoundError:
        pass

    # Get new requirements
    new_requirements = {
        f"{m.project_name}=={m.version}"
        for m in pkg_resources.working_set
        if m.project_name in get_imports() and m.project_name != "pip"
    }

    # Update existing requirements with new ones
    updated_requirements = existing_requirements.union(new_requirements)

    # Write updated requirements to file
    with open(file_path, 'w') as f:
        for requirement in sorted(updated_requirements):
            f.write(f"{requirement}\n")

# Example usage:
update_requirements_file("requirements.txt")
