from glob import glob
from setuptools import setup

package_name = "piracer_state_manager"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adrian",
    maintainer_email="adrian@example.com",
    description="Publishes detector-driven SEARCH, ACQUIRE and FOLLOW behavior state.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "state_manager_node = piracer_state_manager.state_node:main",
            "acquire_node = piracer_state_manager.acquire_node:main",
            "follow_node = piracer_state_manager.follow_node:main",
        ]
    },
)
