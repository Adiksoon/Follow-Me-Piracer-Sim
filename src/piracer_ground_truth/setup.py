from setuptools import find_packages, setup


package_name = 'piracer_ground_truth'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ground_truth.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO:AUTHOR_INPUT',
    maintainer_email='todo-author-input@example.invalid',
    description='Read-only Gazebo model pose publisher for FOLLOW experiments.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ground_truth_publisher = piracer_ground_truth.publisher:main',
        ],
    },
)
