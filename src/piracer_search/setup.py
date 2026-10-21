from glob import glob
from setuptools import setup

package_name = "piracer_search"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/behavior_trees", glob("behavior_trees/*.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adrian",
    maintainer_email="adrian@example.com",
    description="Frontier Search ranking and Nav2 goal adapter for PiRacer.",
    license="MIT",
    entry_points={"console_scripts": [
        "search_node = piracer_search.search_node:main",
        "start_area_node = piracer_search.start_area_node:main",
        "frontier_visualizer = piracer_search.frontier_visualizer:main",
    ]},
)
