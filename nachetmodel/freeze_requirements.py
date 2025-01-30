import subprocess

def get_frozen_packages():
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    frozen_packages = {}
    for line in result.stdout.decode('utf-8').splitlines():
        if '==' in line:
            package, version = line.split('==')
            frozen_packages[package] = version
    return frozen_packages

def read_requirements(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def write_frozen_requirements(requirements, frozen_packages, output_file):
    with open(output_file, 'w') as file:
        for req in requirements:
            package_name = req.split('==')[0]
            if package_name in frozen_packages:
                file.write(f"{package_name}=={frozen_packages[package_name]}\n")

def main():
    frozen_packages = get_frozen_packages()
    requirements = read_requirements('requirements.txt')
    write_frozen_requirements(requirements, frozen_packages, 'requirements-frozen.txt')

if __name__ == "__main__":
    main()