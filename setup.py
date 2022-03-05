from setuptools import setup

setup (
    name="mascots",
    version="0.1",
    packages=['mascots.traceAnalysis', 'mascots.mtCache', 'mascots.blockReader', 'mascots.deviceBenchmark', 'mascots.experiments', 'mascots.dataProcessor'],
    install_requires=["numpy", "pandas", "asserts", "argparse", "scipy"]
)