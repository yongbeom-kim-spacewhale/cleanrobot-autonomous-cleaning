from glob import glob
from setuptools import find_packages, setup

package_name = 'yolo_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', glob('resource/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='habinkang.k@gmail.com',
    description='ROS 2 RGB-D YOLO segmentation node for robot-arm point clouds and AMR targets',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_pub = yolo_pub.yolo_pub:main',
        ],
    },
)
