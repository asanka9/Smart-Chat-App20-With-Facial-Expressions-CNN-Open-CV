# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:03:21 2021

@author: User
"""

import pkg_resources
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
   for i in installed_packages])
for i in installed_packages_list:
    print(i)