from setuptools import setup

package_name = 'piracer_search'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adrian',
    maintainer_email='adrian@example.com',
    description='ROS-independent frontier search core for PiRacer.',
    license='MIT',
)
