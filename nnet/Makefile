#-----------------------------------------------------------------------------
#File Name: Makefile
#
#Purpose: Makefile to simplify nnet build and testing process
#
#Created: 17-Aug-2016 11:18:16 AEST
#-----------------------------------------------------------------------------
#Revision History
#
#
#
#-----------------------------------------------------------------------------
#S.D.G

## LICENSE DETAILS############################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

COVERAGE=--with-coverage --cover-html --cover-package=nnet

test: 
	nosetests $(COVERAGE)  

tests_functional: 
	nosetests $@ $(COVERAGE)

tests_unit: 
	nosetests $@ $(COVERAGE)

build:
	cd ../; python setup.py bdist_wheel; cd -

all: test 

.PHONY: all tests_functional tests_unit clean
