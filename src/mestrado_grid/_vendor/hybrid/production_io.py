"""Install the previously validated atomic-I/O adapter before star imports."""
import atomic_io as io
import common
io.install(common)
import campaign_support
io.install(common, campaign_support)
