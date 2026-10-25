from glob import glob
import os

from setuptools import find_packages, setup


package_name = "piracer_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (
            os.path.join("share", package_name, "behavior_trees"),
            glob("behavior_trees/*.xml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO:AUTHOR_INPUT",
    maintainer_email="todo-author-input@example.invalid",
    description="Minimal PiRacer RGB-D mapping and Nav2 integration.",
    license="MIT",
)
