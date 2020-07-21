# Python Instrument driver for Rhode & Schwarz: NGM202, NGM201, NGL202, NGL201
Requires: pyvisa, numpy
This is an unfinished work in progress.  All is subject to change.  Use at your own risk.
## Basic usage and class flow:
```
import rs_ngm202 as dev          
dev = Device('VISA::ADDRESS')  

(class flow)
Device:         dev  
Display:        dev.display
Log:		dev.log 
Channel:        dev.ch<1,2>
 Arbitary	dev.ch<1,2>.arb
 FastLog	dev.ch<1,2>.flog
 Measure:    	dev.ch<1,2>.meas 
Common:     	dev.com 
Status:     	dev.status
```
### Highlights
* Full control of your NGx device via python
* Log and Fast Log
	* start, stop, configure logging
	* pull existing log data from device into sorted np.arrays
	* Fast Log supports data to both USB or direct to host via SCPI
* Arb
	* Create, edit, and build arb patterns
	* Save created arb pattern, or load an existing one
	* Assign arb patterns to channels