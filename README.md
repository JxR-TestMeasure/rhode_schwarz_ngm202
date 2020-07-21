# Python Instrument driver for Rhode & Schwarz: NGM202, NGM201, NGL202, NGL201
Requires: pyvisa, numpy  
This is an unfinished work in progress.  All is subject to change.  Use at your own risk.
## Basic usage and class flow:
```
import rs_ngm202 as ngm202          
dev = ngm202.Device('VISA::ADDRESS')  

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
## Highlights
* Full control of your NGx device via python
* Log and Fast Log
	* start, stop, configure logging
	* pull existing log data from device into sorted np.arrays
	* Fast Log supports data to both USB or direct to host via SCPI
* Arb
	* Create, edit, and build arb patterns
	* Save created arb pattern, or load an existing one
	* Assign arb patterns to channels

## Usage Examples
### Programming channel output
```
dev.ch1.voltage(5) # 5V
dev.ch1.current(2) # 2A
```
### Get programmed settings
```
dev.ch1.voltage() --> '5.0000000E+00'
dev.ch1.current() --> '2.0000000E+00'
```
### Turn output on
```
dev.ch1.on() 	# turn on ch1
dev.on()	# turn on output
```
or
```
dev.ch1.on_now() # same as the previous commands
```
### FastLog (SCPI)
```
dev.ch1.flog.start_scpi('S500K', 1) 	# Fast log @500kSPS for 0.5s
dev.ch1.flog.start_scpi('S125K', 2) 	# Fast log @125kSPS for 1.0s
```
access data from FastLog(SCPI)
```
dev.ch1.flog.flog_data: dict	<--	# all samples in format np.array(float32)
dev.ch1.flog.flog_data['voltage']	# All voltage samples
dev.ch1.flog.flog_data['current']	# All current samples
dev.ch1.flog.flog_data['power']		# All calculated power samples
dev.ch1.flog.flog_data['seconds']	# calculated time based on sample interval
```
access existing FastLog data on device
```
dev.ch1.flog.get_flog_files()		# gets a list of all '.raw' fastlog files for the channel
dev.ch1.flog.flog_files: dict	<--	# puts them in a dict with key values:(1, 2, 3, .., n)
dev.ch1.flog.build_flog_data(1)		# sorts selected file into np.arrays same as above
```
### Log
```
dev.log.mode('COUN')			# Log data samples to the specified count
dev.log.count(1000)			# log 1000 samples
dev.log.interval(0.1)			# 0.1s sample interval
dev.log.enable()			# Turn logging on
```
access data from log
```
dev.log.get_log_files()			# gets a list of all '.csv' logging files
dev.log.log_files: dict	<--		# puts them in a dict with key values:(1, 2, 3, .., n)
dev.log.build_log_data(1)		# sorts selected file into np.arrays

dev.log.log_data: dict	<--		# all samples in format np.array(float32)
dev.log.log_data['seconds']		# calculated time based on sample interval
dev.log.log_data['voltage_ch1']		# All voltage samples
dev.log.log_data['current_ch1']		# All current samples
dev.log.log_data['power_ch1']		# All calculated power samples
Note: NGx202 devices will also have arrays for ch2
```
### Arbitrary
build a new arb list
```
dev.ch1.arb.clear()  
dev.ch1.arb.add.point(5, 1, 0.1, 1)       # ideally run in a loop  
	.  
	.
dev.ch1.arb.add_point(...)
```
view the arb list
```
dev.ch1.arb.arb_list: dict 	<--	 # All points in :dict with key values:(1, 2, 3, .., n)
dev.ch1.arb.arb_list[1]			 # the first point on the created arb list
```
edit the arb list
```
dev.ch1.arb.edit_point(1024,7,1,3.4,0)	  # Replace point:1024  
```
send the arb list to the device
```
dev.ch1.arb.build()                       # transfer all arb points to device  
```
configure repetition and end behavior
```
dev.ch1.arb.repetitions(2)                # repeat arb 2x **must be after build()
dev.ch1.arb.end_behavior('off')           turn off at end of sequence **must be after build()  
```
save the configured arb sequence
```
dev.ch1.arb.save_to_internal('filename.csv')	# internal storage
dev.ch1.arb.save_to_front_usb('filename.csv')	# front usb storage  
```
load an existing arb sequence
```
dev.ch1.arb.load_from_rear_usb('filename.csv')	# rear usb storage
```
Transfer created arb (or the loaded arb) to the channel
Note: only one transfer command is required
```
dev.ch1.arb.transfer()                    	# transfer arb for channel 1  
dev.ch1.arb.transfer(1)                   	# transfer arb to channel 1    
dev.ch1.arb.transfer(2)                   	# transfer arb to channel 2
```
Enable the arb sequence for the channel and turn channel on
```
dev.ch1.arb.enable()                      enable arb on channel  
dev.ch1.on()
```
### Measurements
All commands output a measured value
```
dev.ch1.meas.voltage() -->	'3.3000212E+00'
dev.ch1.meas.v()		# same as above: v, c, p, cmin, pmax, vavg, etc
dev.ch1.meas.current_max()	
dev.ch1.meas.energy()
dev.ch1.meas.reset_stats()	# reset the min, max, avg statistics
dev.ch1.meas.get_count()	# get current stats counter (count = 0.1s)
etc,
```
