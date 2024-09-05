import sys

from setuptools import setup, find_packages

py_version = sys.version_info[:2]

if py_version < (2, 6):
    raise RuntimeError('On Python 2, this package requires Python 2.6 or later')
elif (3, 0) < py_version < (3, 2):
    raise RuntimeError('On Python 3, this package requires Python 3.2 or later')


install_requires = ['psutil',
                    # 'supervisor>=4.0.0', temporarily disabled
                    ]

if py_version < (3, 2):
    install_requires.append('futures')

tests_require = install_requires + []

setup(
    name='supervisor_checks',
    packages=find_packages(),
    version='0.8.1',
    description='Framework to build health checks for Supervisor-based services.',
    author='Vovan Kuznetsov',
    author_email='vovanec@gmail.com',
    maintainer_email='vovanec@gmail.com',
    url='https://github.com/vovanec/supervisor_checks',
    download_url='https://github.com/vovanec/supervisor_checks/tarball/0.8.1',
    keywords=['supervisor', 'event', 'listener', 'eventlistener',
              'http', 'memory', 'xmlrpc', 'health', 'check', 'monitor', 'cpu'],
    license='MIT',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Development Status :: 4 - Beta',
                 'Intended Audience :: Developers',
                 'Operating System :: POSIX',
                 'Topic :: System :: Boot',
                 'Topic :: System :: Monitoring',
                 'Topic :: System :: Systems Administration',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.6',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.2',
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4'],
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite='nose.collector',
    extras_require={
        'test': tests_require,
    },
    entry_points={
        'console_scripts': [
            'supervisor_memory_check=supervisor_checks.bin.memory_check:main',
            'supervisor_cpu_check=supervisor_checks.bin.cpu_check:main',
            'supervisor_http_check=supervisor_checks.bin.http_check:main',
            'supervisor_tcp_check=supervisor_checks.bin.tcp_check:main',
            'supervisor_xmlrpc_check=supervisor_checks.bin.xmlrpc_check:main',
            'supervisor_complex_check=supervisor_checks.bin.complex_check:main',
            'supervisor_file_check=supervisor_checks.bin.file_check:main']
    }
)

