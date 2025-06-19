from pint import UnitRegistry

# Register units
ureg = UnitRegistry()
ureg.define('Tesla = 1 * tesla = T')
ureg.define('milliTesla = 1e-3 * tesla = mT')
ureg.define('millisecond = 1e-3 * second = ms')