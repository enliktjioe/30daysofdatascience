#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - old tutorial: https://gist.github.com/yawi/0a42269c12ea5238ba03
# 
# Installation:
# 
# - https://www.tutorialspoint.com/apache_storm/apache_storm_installation.htm

# ## Troubleshooting
# ### Zookeeper can't be started
# 
# - run this `sudo bin/zkServer.sh start /opt/apache-zookeeper-3.6.2-bin/conf/zoo.cfg`, source: https://stackoverflow.com/a/31200219
# 
# ### Storm UI can't be started
# - add `ui.port: 8081` to `conf/storm.yaml`, source: https://stackoverflow.com/questions/28473309/storm-ui-not-working



