from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autonomous_vehicles'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='you@example.com',
    description='Pakiet ROS 2 dla pojazdu autonomicznego',
    license='Apache-2.0',
    tests_require=['pytest'],
    
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),

        (os.path.join('share', package_name, 'launch'), ['launch/auto.launch_no_wall.py']),

        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        
        # UWAGA: Usunięto linię instalującą package.xml, aby uniknąć błędu kompilacji!
    ],
    
    entry_points={
        'console_scripts': [
            # Zmieniono na małe litery, aby pasowało do konwencji
            'test_env = autonomous_vehicles.test_env:main', # tutaj trzeba będzie chyba zmienić (zapytać się co będzie wywoływane w )

            # Dodaj swoje pozostałe węzły tutaj, używając małych liter i kropki (np. .nazwa_pliku:main)
        ],
    },
)