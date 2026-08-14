from setuptools import setup

package_name = 'knobbrain'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/backend.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='me439',
    maintainer_email='me439@example.com',
    description='Closed-loop knob turning on top of the xarmrob stack',
    license='MIT',
    entry_points={
        'console_scripts': [
            'brain=knobbrain.brain:main',
            'go=knobbrain.go:main',
        ],
    },
)
