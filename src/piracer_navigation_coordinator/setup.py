from glob import glob
from setuptools import setup


package_name = "piracer_navigation_coordinator"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adrian",
    maintainer_email="adrian@example.com",
    description="Single-owner NavigateToPose action relay for Search and Follow.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "navigation_coordinator_node = "
            "piracer_navigation_coordinator.coordinator_node:main",
        ]
    },
)
