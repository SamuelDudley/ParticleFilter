#!/usr/bin/python
# Upload Card/OS information to Dropbox; called on boot from /etc/rc.local

from subprocess import call, check_output

SCRIPT = "/home/pi/Dropbox-Uploader/dropbox_uploader.sh"
CMD = "upload"

date = check_output(["date"], shell=True)
hostname = check_output(["hostname"], shell=True).strip()
uuid = check_output(["ls -l /dev/disk/by-uuid |grep mmcblk0p1 |cut -c40-48"],
                     shell=True)[:-1]
dpkgs = check_output(["dpkg --get-selections"], shell=True)
pymods = check_output(["pydoc modules"], shell=True)

pymodlist = []
for modline in pymods.splitlines()[3:-4]:
    pymodlist.extend(modline.split())
pymodstr = "\n".join(sorted(pymodlist))

bootcfg = check_output(["cat /boot/config.txt"], shell=True)
rclocal = check_output(["cat /etc/rc.local"], shell=True)
services = check_output(["sudo service --status-all 2>&1"], shell=True)
kmods = check_output(["cat /etc/modules"], shell=True)
lsmod = check_output(["lsmod"], shell=True)
df = check_output(["df"], shell=True)
procvers = check_output(["cat /proc/version"], shell=True)
interfaces = check_output(["cat /etc/network/interfaces"], shell=True)

sep = "\n" + "*" * 79 + "\n"

#filename = "/home/pi/bin/{}.dat".format(uuid)
filename = "/tmp/{}_{}.dat".format(hostname, uuid)
remfilename = "{}_{}.dat".format(hostname, uuid)

fh = open(filename, 'w')
fh.write("Data for card UUID {}\n============================\n".format(uuid))
fh.write("\n{}Output of 'date':{}{}".format(sep, sep, date))
fh.write("\n{}Output of 'hostname':{}{}".format(sep, sep, hostname))
fh.write("\n{}Output of 'cat /proc/version':{}{}".format(sep, sep, procvers))
fh.write("\n{}Output of 'df':{}{}".format(sep, sep, df))
fh.write("\n{}Output of 'cat /etc/network/interfaces':{}{}".format(sep, sep, interfaces))
fh.write("\n{}Output of 'dpkg --get-selections':{}{}".format(sep, sep, dpkgs))
fh.write("\n{}Output of 'pydoc modules':{}{}".format(sep, sep, pymodstr))
fh.write("\n{}Output of 'sudo service --status-all':{}{}".format(sep, sep, services))
fh.write("\n{}Output of 'cat /etc/modules':{}{}".format(sep, sep, kmods))
fh.write("\n{}Output of 'cat /etc/rc.local':{}{}".format(sep, sep, rclocal))
fh.write("\n{}Output of 'lsmod':{}{}".format(sep, sep, lsmod))
fh.close()

check_output(" ".join([SCRIPT, "-f /home/pi/.dropbox_uploader", CMD, filename, remfilename]), shell=True)