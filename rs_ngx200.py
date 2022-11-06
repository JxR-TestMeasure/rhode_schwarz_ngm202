# -*- coding: utf-8 -*-

import pyvisa
import numpy as np
from datetime import datetime
from validation_ngx200 import *
from scpi_instrument_common import Command, Common, ValidateRegister

global_input_values = {}


class Device:
    def __init__(self, visa_addr='TCPIP0::ngm202::inst0::INSTR'):
        self._address = str(visa_addr)
        self._visa_driver = pyvisa.ResourceManager()
        self._bus = self._visa_driver.open_resource(self._address)
        self._bus.read_termination = '\n'
        self._bus.write_termination = '\n'
        self._command = Command(self._bus)

        model = str(self._bus.query('*IDN?')).split(',')[1]
        global_input_values['model'] = model
        global_input_values['ch2'] = '202' in model
        global_input_values['expanded_features'] = 'NGM' in model
        # Device class shortcuts
        self._com = Common(self._bus)
        self.disp = Display(self._bus)
        self.log = Log(self._bus)
        self.trig = Trigger(self._bus)
        self.ch1 = Channel(self._bus, 1)
        if global_input_values['ch2']:
            self.ch2 = Channel(self._bus, 2)

    def write(self, command):
        self._bus.write(command)

    def read(self):
        self._bus.read()

    def query(self, command):
        return self._bus.query(command)

    def read_raw(self):
        return self._bus.read_raw()

    def disconnect(self):
        self._bus.close()

    # set general output on
    def on(self):
        write = 'OUTP:GEN 1'
        self._command.write(write)

    # set general output off
    def off(self):
        write = 'OUTP:GEN 0'
        self._command.write(write)


class Channel:
    def __init__(self, bus, channel):
        self._bus = bus
        self._validate = ValidateChannel()
        self._channel = self._validate.channel(channel)
        if isinstance(self._channel, (ValueError, TypeError)):
            raise self._channel
        self._command = Command(self._bus, self._channel)
        self._chan = {}
        self._chan = {
            'output':   self.is_on(),
            'mode': self.mode(),
            'voltage': self.voltage(),
            'current': self.current(),
            'resistance': self.resistance(),
            'impedance': self.impedance(),
            'resistance_enable': self.get_resistance_state(),
            'impedance_enable': self.get_impedance_state()}
        if global_input_values['expanded_features']:
            self._chan['voltage_range'] = self.voltage_range()
            self._chan['current_range'] = self.current_range()
            self._chan['dvm_enable'] = self.dvm()
        self._chan['ramp_enable'] = self.ramp()
        self._chan['ramp_duration'] = self.ramp_duration()
        self._chan['fast_transient'] = self.fast_transient_response()
        self._chan['output_delay'] = self.output_delay()
        self._chan['output_delay_duration'] = self.output_delay_duration()
        self._chan['trigger_enable'] = self.trigger()
        self._chan['trigger_behavior'] = self.trigger_behavior()
        self.values = {
            'device': global_input_values,
            'settings': self._chan}

        # Channel objects
        self.arb = Arbitrary(self._bus, self._channel)
        if global_input_values['expanded_features']:
            self.flog = FastLog(self._bus, self._channel)
        self.meas = Measure(self._bus, self._channel)
        self.prot = Protection(self._bus, self._channel)

    # ##########################
    # Channel output functions #
    # ##########################

    # Turn channel on (Note: device.on() controls device output)
    def on(self):
        write = 'OUTP:SEL 1'
        self._command.write(write)
        self._chan['output'] = self.is_on()

    # Turn channel and device output on
    def on_now(self):
        write = 'OUTP 1'
        self._command.write(write)
        self._chan['output'] = self.is_on()

    # Turn channel off (Note: device.off() controls device output)
    def off(self):
        write = 'OUTP:SEL 0'
        self._command.write(write)
        self._chan['output'] = self.is_on()

    # get state
    def is_on(self):
        query = 'OUTP?'
        return self._command.read(query)

    # ############################
    # Channel settings functions #
    # ############################

    # Set voltage for channel
    # 0-20V @ 1mV resolution
    def voltage(self, set_voltage=None):
        query = 'VOLT?'
        write = 'VOLT'
        return self._command.read_write(
            query, write, self._validate.voltage,
            set_voltage, self._chan, 'voltage')

    # Set current limit for channel
    # 0.1mA-6A @ 0-6V; 0.1mA-3A @6-20V; resolution 0.1mA
    def current(self, set_current=None):
        query = 'CURR?'
        write = 'CURR'
        return self._command.read_write(
            query, write, self._validate.current,
            set_current, self._chan, 'current')

    # Set voltage range for channel
    # auto, 5V, 20V ranges
    # Valid Input: 'auto', 5, 20
    def voltage_range(self, set_voltage_range=None):
        if global_input_values['expanded_features']:
            query = 'SENS:VOLT:RANG?'
            write = 'SENS:VOLT:RANG'
            if str(set_voltage_range).upper() == 'AUTO':
                write = 'SENS:VOLT:RANG:AUTO 1'
                self._command.write(write)
                self._chan['voltage_range'] = self.voltage_range()
                return None
            return self._command.read_write(
                query, write, self._validate.voltage_range,
                set_voltage_range, self._chan, 'voltage_range')

    # Set current range for channel
    # auto, 10A, 1A, 100mA, 10mA ranges
    # Valid Input: 'auto', 10, 1, 0.1, 0.01
    def current_range(self, set_current_range=None):
        if global_input_values['expanded_features']:
            query = 'SENS:CURR:RANG?'
            write = 'SENS:CURR:RANG'
            if str(set_current_range).upper() == 'AUTO':
                write = 'SENS:CURR:RANG:AUTO 1'
                self._command.write(write)
                self._chan['current_range'] = self.current_range()
                return None
            return self._command.read_write(
                query, write, self._validate.current_range,
                set_current_range, self._chan, 'current_range')

    # Set output mode for channel
    # auto - quadrant 1 or 2 operation
    # sink - quadrant 2 operation
    # source - quadrant 1 operation
    # Valid input: 'auto', 'sink', 'source'
    def mode(self, set_mode=None):
        query = 'OUTP:MODE?'
        write = 'OUTP:MODE'
        return self._command.read_write(
            query, write, self._validate.mode,
            set_mode, self._chan, 'mode')

    # Set resistance for channel (CR mode)
    # 0-10kOhm; resolution 1mOhm
    def resistance(self, set_resistance=None):
        query = 'RES?'
        write = 'RES'
        return self._command.read_write(
            query, write, self._validate.resistance,
            set_resistance, self._chan, 'resistance')

    # Set CR mode on for channel
    def resistance_on(self):
        write = 'RES:STAT 1'
        self._command.write(write)

    # Set CR mode off for channel
    def resistance_off(self):
        write = 'RES:STAT 0'
        self._command.write(write)

    def get_resistance_state(self):
        query = 'RES:STAT?'
        return self._command.read(query)

    # Set output impedance for channel (Battery simulation)
    # -0.05 - 100 Ohms; 1 mOhm resolution
    def impedance(self, set_impedance=None):
        query = 'OUTP:IMP?'
        write = 'OUTP:IMP'
        return self._command.read_write(
            query, write, self._validate.impedance,
            set_impedance, self._chan, 'impedance')

    def impedance_on(self):
        write = 'OUTP:IMP:STAT 1'
        self._command.write(write)
        self._chan['impedance_enable'] = self.get_impedance_state()

    def impedance_off(self):
        write = 'OUTP:IMP:STAT 0'
        self._command.write(write)
        self._chan['impedance_enable'] = self.get_impedance_state()

    def get_impedance_state(self):
        query = 'OUTP:IMP:STAT?'
        return self._command.read(query)

    def fast_transient_response(self, set_fast_transient_response=None):
        query = 'OUTP:FTR?'
        write = 'OUTP:FTR'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_fast_transient_response, self._chan, 'fast_transient_response')

    def ramp(self, set_ramp_on_off=None):
        query = 'VOLT:RAMP?'
        write = 'VOLT:RAMP'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_ramp_on_off, self._chan, 'ramp_enable')

    def ramp_duration(self, set_ramp_duration=None):
        query = 'VOLT:RAMP:DUR?'
        write = 'VOLT:RAMP:DUR'
        return self._command.read_write(
            query, write, self._validate.ramp_duration,
            set_ramp_duration, self._chan, 'ramp_duration')

    def dvm(self, set_dvm_on_off=None):
        query = 'VOLT:DVM?'
        write = 'VOLT:DVM'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_dvm_on_off, self._chan, 'dvm_enable')

    def output_delay(self, set_output_delay=None):
        query = 'OUTP:DEL?'
        write = 'OUTP:DEL'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_output_delay, self._chan, 'output_delay')

    def output_delay_duration(self, set_output_delay_duration=None):
        query = 'OUTP:DEL:DUR?'
        write = 'OUTP:DEL:DUR'
        return self._command.read_write(
            query, write, self._validate.output_delay_duration,
            set_output_delay_duration, self._chan, 'output_delay_duration')

    def trigger(self, set_trigger_on_off=None):
        query = 'OUTP:TRIG?'
        write = 'OUTP:TRIG'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_trigger_on_off, self._chan, 'trigger_enable')

    def trigger_behavior(self, set_output_behavior=None):
        query = 'OUTP:TRIG:BEH?'
        write = 'OUTP:TRIG:BEH'
        return self._command.read_write(
            query, write, self._validate.trigger_behavior,
            set_output_behavior, self._chan, 'trigger_output_behavior')


class Display:
    def __init__(self, bus):
        self._bus = bus
        self._validate = ValidateDisplay()
        self._command = Command(self._bus)
        self._disp = {}
        self._disp = {'brightness': self.brightness()}
        self.values = {
            'device': global_input_values,
            'settings': self._disp}

    def brightness(self, set_brightness=None):
        query = 'DISP:BRIG?'
        write = 'DISP:BRIG'
        return self._command.read_write(
            query, write, self._validate.brightness,
            set_brightness, self._disp, 'brightness')

    def set_message(self, message):
        write = 'DISP:TEXT "' + str(message) + '"'
        self._command.write(write)

    def clear_message(self):
        write = 'DISP:CLE'
        self._command.write(write)


# 'enable' key does not track when log stopped after starting (fix unlikely)
# @TODO: error checking for log data, more file commands, add some metadata from log file

class Log:
    def __init__(self, bus):
        self._bus = bus
        self._command = Command(self._bus)
        self._validate = ValidateLog()
        self.log_files = {}
        self.log_data = {}
        self._log = {}
        self._log = {
            'enable': self.get_enable(),
            'mode': self.mode(),
            'count': self.count(),
            'duration': self.duration(),
            'interval': self.interval(),
            'file_name': self.get_file_name(),
            'start_time': self.get_start_time(),
            'trigger_enable': self.trigger()}
        self.values = {
            'device': global_input_values,
            'settings': self._log}

    def disable(self):
        # Direct pyvisa call was required here?
        self._command.write('LOG 0')
        self._log['enable'] = self.get_enable()

    def enable(self):
        # Direct pyvisa call was required here?
        self._command.write('LOG 1')
        self._log['enable'] = self.get_enable()

    def get_enable(self):
        query = 'LOG?'
        return self._command.read(query)

    def mode(self, set_mode=None):
        query = 'LOG:MODE?'
        write = 'LOG:MODE'
        return self._command.read_write(
            query, write, self._validate.mode,
            set_mode, self._log, 'mode')

    def count(self, set_count=None):
        query = 'LOG:COUN?'
        write = 'LOG:COUN'
        return self._command.read_write(
            query, write, self._validate.count,
            set_count, self._log, 'count')

    def duration(self, set_duration=None):
        query = 'LOG:COUN?'
        write = 'LOG:COUN'
        return self._command.read_write(
            query, write, self._validate.duration,
            set_duration, self._log, 'duration')

    def interval(self, set_interval=None):
        query = 'LOG:INT?'
        write = 'LOG:INT'
        return self._command.read_write(
            query, write, self._validate.interval,
            set_interval, self._log, 'interval')

    def get_file_name(self):
        query = 'LOG:FNAM?'
        return self._command.read(query)

    # argument must be of class datetime from module datetime
    # no further validation will be performed...
    def set_start_time(self, date_and_time: datetime):
        write = 'LOG:STIM '\
                + str(date_and_time.year) + ','\
                + str(date_and_time.month) + ','\
                + str(date_and_time.day) + ','\
                + str(date_and_time.hour) + ','\
                + str(date_and_time.minute) + ','\
                + str(date_and_time.second)
        # Direct pyvisa call was required here?
        self._command.write(write)
        self._log['start_time'] = self.get_start_time()

    def get_start_time(self):
        query = 'LOG:STIM?'
        return self._command.read(query)

    def get_log_files(self):
        self.log_files.clear()
        key_count = 0
        query = 'DATA:LIST?'
        all_file_paths = str(self._command.read(query)).split(',')
        for file_path in all_file_paths:
            if '/logging/' in file_path:
                key_count += 1
                self.log_files[key_count] = file_path

    def build_log_data(self, log_file_key: int):
        query = 'DATA:DATA? ' + self.log_files[log_file_key]
        result = self._command.read(query)
        result_array = result.split('\r\n')
        meas_array = result_array[17:-1]
        interval = float(result_array[6].split(',')[1])
        number_samples = result_array.__len__() - 18
        self.log_data['seconds'] = np.arange(0, interval * number_samples, interval, dtype='f').round(1)
        v1 = []
        i1 = []
        p1 = []
        v2 = []
        i2 = []
        p2 = []
        for samples_text in meas_array:
            samples = samples_text.split(',')
            v1.append(np.single(samples[1]))
            i1.append(np.single(samples[2]))
            p1.append(np.single(samples[3]))
            if global_input_values['ch2']:
                v2.append(np.single(samples[4]))
                i2.append(np.single(samples[5]))
                p2.append(np.single(samples[6]))
        self.log_data['voltage_ch1'] = np.array(v1).round(6)
        self.log_data['current_ch1'] = np.array(i1).round(8)
        self.log_data['power_ch1'] = np.array(p1).round(6)
        if global_input_values['ch2']:
            self.log_data['voltage_ch2'] = np.array(v2).round(6)
            self.log_data['current_ch2'] = np.array(i2).round(8)
            self.log_data['power_ch2'] = np.array(p2).round(6)

    def delete_log_file(self, log_file_key: int):
        write = 'DATA:DEL ' + self.log_files[log_file_key]
        self._command.write(write)
        self.get_log_files()

    def trigger(self, set_trigger_on_off=None):
        query = 'OUTP:TRIG?'
        write = 'OUTP:TRIG'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_trigger_on_off, self._log, 'trigger_enable')


# @TODO: Why does internal FLOG and SCPI flog produce different data sizes for same sample time??
# 1s S500K internal fast log = 500,000 samples of V/I
# 1s S500K scpi fast log = 524,160 samples of V/I
class FastLog:
    def __init__(self, bus, channel):
        self._bus = bus
        self._validate = ValidateFastLog()
        self._channel = channel
        self._command = Command(self._bus, self._channel)
        self._flog_enable = False if self.get_enable() == '0' else True
        self._flog = {}
        self._flog = {
            'enable': self._flog_enable,
            'sample_rate': self.sample_rate(),
            'sample_interval': '0',
            'trigger_enable': self.trigger()}
        self.values = {
            'device': global_input_values,
            'settings': self._flog}
        self.flog_data = {}
        self.flog_files = {}

        self.com = Common(self._bus)
        self.status = Status(self._bus, self._channel)

    # @TODO: file management, flog config methods

    def disable(self):
        write = ':FLOG:STAT 0'
        self._command.write(write)
        self._flog_enable = False

    def enable(self):
        self.com.wait()
        write = ':FLOG:STAT 1'
        self._command.write(write)
        self._flog_enable = True

    def get_enable(self):
        query = ':FLOG?'
        return self._command.read(query)

    # @TODO: Command 'FLOG:TARG' undocumented by R&S
    def target_usb(self):
        write = 'FLOG:TARG USB'
        self._command.write(write)

    # @TODO: Command 'FLOG:TARG' undocumented by R&S
    def target_scpi(self):
        write = ':FLOG:TARG SCPI'
        self._command.write(write)

    # @TODO: Command 'FLOG:WFIL' does not work as documented by R&S
    #def local_file_location(self, value: str):
    #    write = 'FLOG:WFIL'
    #    self._command.write_value(write, None, value)

    def sample_rate(self, set_sample_rate=None):
        query = ':FLOG:SRAT?'
        write = ':FLOG:SRAT'
        return self._command.read_write(
            query, write, self._validate.sample_rates,
            set_sample_rate, self._flog, 'sample_rate')

    # Populated flog_file:dict with raw flog file paths on inserted USB(s)
    def get_flog_files(self):
        self.flog_files.clear()
        chan_raw = 'ch' + self._channel + '.raw'
        key_count = 0
        query = 'DATA:LIST?'
        all_file_paths = str(self._command.read(query)).split(',')
        for file_path in all_file_paths:
            if '/fastlog/' in file_path:
                if chan_raw in file_path:
                    key_count += 1
                    self.flog_files[key_count] = file_path

    # Import raw flog data into np.arrays
    # Pass the dictionary key for the file you want to import from flog_files:dict
    # example: build_flog_data(1)
    def build_flog_data(self, flog_file_key: int):
        raw_file = str(self.flog_files[flog_file_key])
        meta_file = raw_file.replace('.raw', '.meta')
        write = 'DATA:DATA? ' + raw_file
        self._command.write(write)
        raw_data = self._bus.read_raw()
        query = 'DATA:DATA? ' + meta_file
        meta_data = self._command.read(query)
        sample_rate = float(meta_data.split('\n')[5].split('\t')[1])
        interval = 1 / sample_rate
        data = np.frombuffer(raw_data[0:-1], dtype='<f4')
        number_samples = data.size / 2
        self._flog['sample_rate'] = sample_rate
        self._flog['sample_interval'] = interval
        self.flog_data['seconds'] = np.arange(0, interval * number_samples, interval, dtype='f').round(7)
        self.flog_data['voltage'] = np.array(data[0::2]).round(6)
        self.flog_data['current'] = np.array(data[1::2]).round(8)
        self.flog_data['power'] = np.array(self.flog_data['voltage'] * self.flog_data['current']).round(6)

    def delete_flog_file(self, flog_file_key: int):
        raw_file = self.flog_files[flog_file_key]
        meta_file = raw_file.replace('.raw', '.meta')
        write = 'DATA:DEL ' + raw_file
        self._command.write(write)
        write = 'DATA:DEL ' + meta_file
        self._command.write(write)
        self.get_flog_files()

    # @TODO Does not work as intended due to undocumented bugs in SCPI command 'FLOG:WFIL'
    # Duration must be set manually on front panel
    def start_local(self, sample_rate='S15', sample_duration=0.1):
        self.disable()
        self.sample_rate(sample_rate)
        self.target_usb()
        # write = 'FLOG:WFIL 1'       # SCPI Error (seems redundent with 'FLOG:TARG USB'
        # self._command.write(write)
        # write = 'FLOG:WFIL:DUR'     # SCPI Error
        # self._command.write_value(write, self._validate.write_duration, sample_duration)
        self.enable()

    def initialize_scpi(self):
        self.disable()
        # Clear event registers
        self.status.get_opr_inst_event_reg()
        self.status.get_opr_inst_sum_event_reg()
        # something
        self.status.opr_enable_reg(8192)
        self.status.opr_ptr_reg(8192)
        self.status.opr_ntr_reg(0)
        # something
        self.status.opr_inst_enable_reg(7)
        self.status.opr_inst_ptr_reg(7)
        self.status.opr_inst_ntr_reg(0)
        # clear event registers
        self.status.get_opr_event_reg()
        self.status.get_opr_inst_event_reg()
        self.target_scpi()

    # @TODO: Sample time unconfirmed by R&S
    # Turn on fast logging for the channel
    # All fast log samples are in blocks of 0.5s
    #   num_samples = 2 --> fast log for 1.0s
    # Sample rates:
    #       'S15'     :   ~16  SPS
    #       'S30'     :   ~32  SPS
    #       'S61'     :   ~64  SPS
    #       'S122'    :   ~128 SPS
    #       'S244'    :   ~256 SPS
    #       'S488'    :   ~512 SPS
    #       'S976'    :     ~1 kSPS
    #       'S1K953'  :     ~2 kSPS
    #       'S3K906'  :     ~4 kSPS
    #       'S7K812'  :     ~8 kSPS
    #       'S15K625' :    ~16 kSPS
    #       'S31K25'  :    ~32 kSPS
    #       'S62K5'   :    ~64 kSPS
    #       'S125K'   :   ~128 kSPS
    #       'S250K'   :   ~256 kSPS
    #       'S500K'   :   ~525 kSPS
    def start_scpi(self, sample_rate='S15', num_samples=1):
        counter = num_samples
        inst_channel_bit = 2 if self._channel == '1' else 4
        stb_opr_status_bit = 128
        flog_data_rdy_bit = 4096
        opr_event_bit = 8192
        self.initialize_scpi()
        self.sample_rate(sample_rate)
        self.enable()
        while self._flog_enable:
            status_byte_reg = int(self._bus.read_stb())
            if status_byte_reg & stb_opr_status_bit:
                opr_event_reg = int(self.status.get_opr_event_reg())
                if opr_event_reg & opr_event_bit:
                    inst_event_reg = int(self.status.get_opr_inst_event_reg())
                    if inst_event_reg & inst_channel_bit:
                        inst_sum_event_reg = int(self.status.get_opr_inst_sum_event_reg())
                        if inst_sum_event_reg & flog_data_rdy_bit:
                            self._command.set_channel()
                            data = np.array(
                                    self._bus.query_binary_values(':FLOG:DATA?'), dtype='f')
                            np.set_printoptions(
                                    formatter={'float': '{:.6e}'.format}, suppress=False)
                            if counter == num_samples:
                                self.flog_data['voltage'] = np.array(data[0::2]).round(6)
                                self.flog_data['current'] = np.array(data[1::2]).round(8)
                            else:
                                self.flog_data['voltage'] = np.append(
                                    self.flog_data['voltage'], data[0::2], axis=0).round(6)
                                self.flog_data['current'] = np.append(
                                    self.flog_data['current'], data[1::2], axis=0).round(8)
                            counter -= 1
                            if counter == 0:
                                self.disable()
        self.flog_data['power'] = np.array(self.flog_data['voltage'] * self.flog_data['current']).round(6)
        self._flog['sample_time'] = np.single(
                (float(num_samples) * 0.5) / self.flog_data['voltage'].size)
        self.flog_data['seconds'] = np.arange(
                0, self._flog['sample_time'] * self.flog_data['voltage'].size,
                self._flog['sample_time'], dtype='d').round(9)

    def trigger(self, set_trigger_on_off=None):
        query = 'OUTP:TRIG?'
        write = 'OUTP:TRIG'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_trigger_on_off, self._flog, 'trigger_enable')


class Arbitrary:
    def __init__(self, bus, channel: str):
        self._bus = bus
        self._channel = channel
        self._validate = ValidateArbitrary()
        self._command = Command(self._bus, self._channel)
        self._arb_count = 0
        self.arb_list = {}
        self._arb = {}
        self._arb = {'enable': self.get_enable(),
                     'points': self._arb_count,
                     'trigger_enable': self.trigger(),
                     'trigger_mode': self.trigger_mode()}
        self.values = {
            'device': global_input_values,
            'settings': self._arb}

    def disable(self):
        write = ':ARB 0'
        self._command.write(write)
        self._arb['enable'] = self.get_enable()

    def enable(self):
        write = ':ARB 1'
        self._command.write(write)
        self._arb['enable'] = self.get_enable()

    def get_enable(self):
        query = ':ARB?'
        return self._command.read(query)

    def clear(self):
        self.arb_list.clear()
        self._arb_count = 0
        self._arb['points'] = self._arb_count
        write = 'ARB:CLE'
        self._command.write(write)

    def add_point(self, voltage, current, dwell_time, interpolation):
        arb_data = {}
        arb_data['voltage'] = voltage
        arb_data['current'] = current
        arb_data['dwell_time'] = dwell_time
        arb_data['interpolation'] = interpolation
        if self._arb_count < 4096:
            self._arb_count += 1
            self.arb_list[self._arb_count] = arb_data
            self._arb['points'] = self._arb_count
        else:
            print('Maximum arb points reached!')

    def edit_point(self, point: int, voltage, current, dwell_time, interpolation):
        if 1 <= point <= self._arb_count:
            arb_data = {}
            arb_data['voltage'] = voltage
            arb_data['current'] = current
            arb_data['dwell_time'] = dwell_time
            arb_data['interpolation'] = interpolation
            self.arb_list[point] = arb_data
        else:
            print('Arb point not found!')

    def build(self):
        if self._validate.arb_list(self.arb_list):
            arb_data = ''
            for x in self.arb_list.keys():
                arb_data += str(self.arb_list[x]['voltage']) + ','
                arb_data += str(self.arb_list[x]['current']) + ','
                arb_data += str(self.arb_list[x]['dwell_time']) + ','
                arb_data += str(self.arb_list[x]['interpolation']) + ','
            write = 'ARB:DATA ' + arb_data[0:-1]
            self._command.write(write)
        else:
            print('Unable to build arb list')

    def transfer(self, channel=None):
        if channel is None:
            write = 'ARB:TRAN ' + self._channel
            self._command.write(write)
        else:
            write = 'ARB:TRAN'
            self._command.write_value(write, self._validate.channel, channel)

    def repetitions(self, set_num_repeats=None):
        query = 'ARB:REP?'
        write = 'ARB:REP'
        return self._command.read_write(
            query, write, self._validate.repetition,
            set_num_repeats, self._arb, 'repetitions')

    def end_behavior(self, set_end_behavior=None):
        query = 'ARB:BEH:END?'
        write = 'ARB:BEH:END'
        return self._command.read_write(
            query, write, self._validate.end_behavior,
            set_end_behavior, self._arb, 'end_behavior')

    def save_to_internal(self, file_name_csv: str):
        write = 'ARB:FNAME "' + file_name_csv + '", INT'
        self._command.write(write)
        write = 'ARB:SAVE'
        self._command.write(write)

    # You must still use transfer() to activate on channel
    def load_from_internal(self, file_name_csv: str):
        write = 'ARB:FNAME "' + file_name_csv + '", INT'
        self._command.write(write)
        write = 'ARB:LOAD'
        self._command.write(write)

    def save_to_front_usb(self, file_name_csv: str):
        write = 'ARB:FNAME "'\
                + '/USB1A/'\
                + global_input_values['model']\
                + '/arb/'\
                + file_name_csv + '", EXT'
        self._command.write(write)
        write = 'ARB:SAVE'
        self._command.write(write)

    def save_to_rear_usb(self, file_name_csv: str):
        write = 'ARB:FNAME "'\
                + '/USB2A/'\
                + global_input_values['model']\
                + '/arb/'\
                + file_name_csv + '", EXT'
        self._command.write(write)
        write = 'ARB:SAVE'
        self._command.write(write)

    # You must still use transfer() to activate on channel
    def load_from_front_usb(self, file_name_csv: str):
        write = 'ARB:FNAME "'\
                + '/USB1A/'\
                + global_input_values['model']\
                + '/arb/'\
                + file_name_csv + '", EXT'
        self._command.write(write)
        write = 'ARB:LOAD'
        self._command.write(write)

    # You must still use transfer() to activate on channel
    def load_from_rear_usb(self, file_name_csv: str):
        write = 'ARB:FNAME "'\
                + '/USB2A/'\
                + global_input_values['model']\
                + '/arb/'\
                + file_name_csv + '", EXT'
        self._command.write(write)
        write = 'ARB:LOAD'
        self._command.write(write)

    def trigger(self, set_trigger_on_off=None):
        query = 'OUTP:TRIG?'
        write = 'OUTP:TRIG'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_trigger_on_off, self._arb, 'trigger_enable')

    def trigger_mode(self, set_trigger_mode=None):
        query = 'ARB:TRIG:MODE?'
        write = 'ARB:TRIG:MODE'
        return self._command.read_write(
            query, write, self._validate.trigger_mode,
            set_trigger_mode, self._arb, 'trigger_mode')


class Measure:
    def __init__(self, bus, channel):
        self._bus = bus
        self._channel = channel
        self._command = Command(self._bus, self._channel)

    def __get_stat(self, meas_source: str, stat_type: str):
        query = 'MEAS:' + meas_source + ':' + stat_type + '?'
        return self._command.read(query)

    # ###############################
    # Channel measurement functions #
    # ###############################

    # #########
    # Voltage #
    # #########

    def voltage(self):
        query = 'MEAS:VOLT?'
        return self._command.read(query)

    def voltage_avg(self):
        return self.__get_stat('VOLT', 'AVG')

    def voltage_min(self):
        return self.__get_stat('VOLT', 'MIN')

    def voltage_max(self):
        return self.__get_stat('VOLT', 'MAX')

    def v(self):
        return self.voltage()

    def vavg(self):
        return self.voltage_avg()

    def vmax(self):
        return self.voltage_max()

    def vmin(self):
        return self.voltage_min()

    # #########
    # Current #
    # #########

    def current(self):
        query = 'MEAS:CURR?'
        return self._command.read(query)

    def current_avg(self):
        return self.__get_stat('CURR', 'AVG')

    def current_min(self):
        return self.__get_stat('CURR', 'MIN')

    def current_max(self):
        return self.__get_stat('CURR', 'MAX')

    def c(self):
        return self.current()

    def cavg(self):
        return self.current_avg()

    def cmax(self):
        return self.current_max()

    def cmin(self):
        return self.current_min()

    # ########
    #  Power #
    # ########

    def power(self):
        query = 'MEAS:POW?'
        return self._command.read(query)

    def power_avg(self):
        return self.__get_stat('POW', 'AVG')

    def power_min(self):
        return self.__get_stat('POW', 'MIN')

    def power_max(self):
        return self.__get_stat('POW', 'MAX')

    def p(self):
        return self.power()

    def pavg(self):
        return self.power_avg()

    def pmax(self):
        return self.power_max()

    def pmin(self):
        return self.power_min()

    # #########
    #  Energy #
    # #########

    def energy(self):
        query = 'MEAS:ENER?'
        return self._command.read(query)

    def reset_energy(self):
        write = ':MEAS:ENER:RES'
        self._command.write(write)

    # ###############
    # Stats Control #
    # ###############

    def reset_stats(self):
        write = ':MEAS:STAT:RES'
        self._command.write(write)

    # 1 count = 0.1s
    def get_stats_count(self):
        query = ':MEAS:STAT:COUN?'
        return self._command.read(query)


# @TODO not implemented
class Battery:
    def __init__(self, bus):
        self._bus = bus
    pass


# @TODO safety limits
class Protection:
    def __init__(self, bus, channel):
        self._bus = bus
        self._channel = channel
        self._command = Command(self._bus, self._channel)
        self._validate = ValidateProtection()
        self._prot = {}
        self._prot = {
            'ocp_enable': self.ocp(),
            'ovp_enable': self.ovp(),
            'opp_enable': self.opp(),
            'ocp_delay_initial': self.ocp_delay_initial(),
            'ocp_delay': self.ocp_delay(),
            'ocp_link': self.get_ocp_link(),
            'ovp_level': self.ovp_level(),
            'opp_level': self.opp_level(),
            'limits_enable': self.limits(),
            'limits_voltage_low': self.limits_voltage_low(),
            'limits_voltage_high': self.limits_voltage_high(),
            'limits_current_low': self.limits_current_low(),
            'limits_current_high': self.limits_current_high()}
        self.values = {
            'device': global_input_values,
            'settings': self._prot}

    def ocp(self, set_ocp_state=None):
        query = 'CURR:PROT?'
        write = 'CURR:PROT'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_ocp_state, self._prot, 'ocp_enable')

    def ocp_clear(self):
        write = 'CURR:PROT:CLE'
        self._command.write(write)

    def ocp_delay_initial(self, set_initial_delay=None):
        query = 'CURR:PROT:DEL?'
        write = 'CURR:PROT:DEL'
        return self._command.read_write(
            query, write, self._validate.delay_initial,
            set_initial_delay, self._prot, 'ocp_delay_initial')

    def ocp_delay(self, set_delay=None):
        query = 'CURR:PROT:DEL:INIT?'
        write = 'CURR:PROT:DEL:INIT'
        return self._command.read_write(
            query, write, self._validate.delay,
            set_delay, self._prot, 'ocp_delay')

    def ocp_link(self):
        other_channel = '2' if self._channel == '1' else '1'
        write = 'CURR:PROT:LINK ' + other_channel
        self._command.write(write)
        self._prot['ocp_link'] = self.get_ocp_link()

    def ocp_unlink(self):
        other_channel = '2' if self._channel == '1' else '1'
        write = 'CURR:PROT:UNL ' + other_channel
        self._command.write(write)
        self._prot['ocp_link'] = self.get_ocp_link()

    def get_ocp_link(self):
        other_channel = '2' if self._channel == '1' else '1'
        query = 'CURR:PROT:LINK? ' + other_channel
        return self._command.read(query)

    def get_ocp_trip(self):
        query = 'CURR:PROT:TRIP?'
        self._command.read(query)

    def ovp(self, set_ovp_state=None):
        query = 'VOLT:PROT?'
        write = 'VOLT:PROT'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_ovp_state, self._prot, 'ovp_enable')

    def ovp_clear(self):
        write = 'VOLT:PROT:CLE'
        self._command.write(write)

    def ovp_level(self, set_ovp_level=None):
        query = 'VOLT:PROT:LEV?'
        write = 'VOLT:PROT:LEV'
        return self._command.read_write(
            query, write, self._validate.voltage,
            set_ovp_level, self._prot, 'ovp_level')

    def get_ovp_trip(self):
        query = 'VOLT:PROT:TRIP?'
        self._command.read(query)

    def opp(self, set_opp_state=None):
        query = 'POW:PROT?'
        write = 'POW:PROT'
        return self._command.read_write(
            query, write, self._validate.on_off,
            set_opp_state, self._prot, 'opp_enable')

    def opp_clear(self):
        write = 'POW:PROT:CLE'
        self._command.write(write)

    def opp_level(self, set_opp_level=None):
        query = 'POW:PROT:LEV?'
        write = 'POW:PROT:LEV'
        return self._command.read_write(
            query, write, self._validate.power,
            set_opp_level, self._prot, 'opp_level')

    def get_opp_trip(self):
        query = 'POW:PROT:TRIP?'
        self._command.read(query)

    def limits(self, safety_limits_on_off=None):
        query = 'ALIM?'
        write = 'ALIM'
        return self._command.read_write(
            query, write, self._validate.on_off,
            safety_limits_on_off, self._prot, 'limits_enable')

    def limits_voltage_low(self, set_lower_voltage_lim=None):
        query = 'VOLT:ALIM:LOW?'
        write = 'VOLT:ALIM:LOW'
        return self._command.read_write(
            query, write, self._validate.voltage,
            set_lower_voltage_lim, self._prot, 'limits_voltage_low')

    def limits_voltage_high(self, set_upper_voltage_lim=None):
        query = 'VOLT:ALIM?'
        write = 'VOLT:ALIM'
        return self._command.read_write(
            query, write, self._validate.voltage,
            set_upper_voltage_lim, self._prot, 'limits_voltage_high')

    def limits_current_low(self, set_lower_current_lim=None):
        query = 'CURR:ALIM:LOW?'
        write = 'CURR:ALIM:LOW'
        return self._command.read_write(
            query, write, self._validate.voltage,
            set_lower_current_lim, self._prot, 'limits_current_low')

    def limits_current_high(self, set_upper_current_lim=None):
        query = 'CURR:ALIM?'
        write = 'CURR:ALIM'
        return self._command.read_write(
            query, write, self._validate.voltage,
            set_upper_current_lim, self._prot, 'limits_current_high')


# @TODO not implemented
class DigitalIO:
    def __init__(self, bus, channel):
        self._bus = bus
        self._channel = channel
    pass


class Trigger:
    def __init__(self, bus):
        self._bus = bus
        self._command = Command(self._bus)
        self._validate = ValidateTrigger()
        self._trig = {}
        self._trig = {
            'enable': self.get_enable(),
            'source': self.source(),
            'dio_channel': self.dio_channel(),
            'dio_pin': self.dio_pin(),
            'output_mode': self.output_mode(),
            'output_mode_channel': self.output_mode_channel(),
            'output_channel': self.output_channel()}
        self.values = {
            'device': global_input_values,
            'settings': self._trig}

    def enable(self):
        write = 'TRIG 1'
        self._command.write(write)
        self._trig['enable'] = self.get_enable()

    def disable(self):
        write = 'TRIG 0'
        self._command.write(write)
        self._trig['enable'] = self.get_enable()

    def get_enable(self):
        query = 'TRIG?'
        return self._command.read(query)

    def source(self, set_source=None):
        query = 'TRIG:SOUR?'
        write = 'TRIG:SOUR'
        return self._command.read_write(
            query, write, self._validate.source,
            set_source, self._trig, 'source')

    def dio_channel(self, set_dio_chan=None):
        query = 'TRIG:SOUR:DIO:CHAN?'
        write = 'TRIG:SOUR:DIO:CHAN'
        return self._command.read_write(
            query, write, self._validate.channel,
            set_dio_chan, self._trig, 'dio_channel')

    def dio_pin(self, set_io_pin=None):
        query = 'TRIG:SOUR:DIO:PIN?'
        write = 'TRIG:SOUR:DIO:PIN'
        return self._command.read_write(
            query, write, self._validate.pin,
            set_io_pin, self._trig, 'dio_pin')

    def output_mode(self, set_output_mode=None):
        query = 'TRIG:SOUR:OMOD?'
        write = 'TRIG:SOUR:OMOD'
        return self._command.read_write(
            query, write, self._validate.output_mode,
            set_output_mode, self._trig, 'output_mode')

    def output_mode_channel(self, set_output_mode_channel=None):
        query = 'TRIG:SOUR:OMOD:CHAN?'
        write = 'TRIG:SOUR:OMOD:CHAN'
        return self._command.read_write(
            query, write, self._validate.channel,
            set_output_mode_channel, self._trig, 'output_mode_channel')

    def output_channel(self, set_output_channel=None):
        query = 'TRIG:SOUR:OUTP:CHAN?'
        write = 'TRIG:SOUR:OUTP:CHAN'
        return self._command.read_write(
            query, write, self._validate.channel,
            set_output_channel, self._trig, 'output_channel')


class Status:
    def __init__(self, bus, channel: str):
        self._bus = bus
        self._validate = ValidateRegister()
        self._channel = channel
        self._command = Command(self._bus)
        self.com = Common(self._bus)

    # @TODO: Register layout and commands undocumented by R&S ....
    # ############################
    # # Operation event register #
    # ############################

    def get_opr_cond_reg(self):
        query = 'STAT:OPER:COND?'
        return self._command.read(query)

    def get_opr_event_reg(self):
        query = 'STAT:OPER:EVEN?'
        return self._command.read(query)

    def opr_enable_reg(self, reg_value=None):
        query = 'STAT:OPER:ENAB?'
        write = 'STAT:OPER:ENAB'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def opr_ptr_reg(self, reg_value=None):
        query = 'STAT:OPER:PTR?'
        write = 'STAT:OPER:PTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def opr_ntr_reg(self, reg_value=None):
        query = 'STAT:OPER:NTR?'
        write = 'STAT:OPER:NTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    # @TODO: Register layout undocumented by R&S ....
    # #######################################
    # # Operation event instrument register #
    # #######################################

    def get_opr_inst_cond_reg(self):
        query = 'STAT:OPER:INST:COND?'
        return self._command.read(query)

    def get_opr_inst_event_reg(self):
        query = 'STAT:OPER:INST:EVEN?'
        return self._command.read(query)

    def opr_inst_enable_reg(self, reg_value=None):
        query = 'STAT:OPER:INST:ENAB?'
        write = 'STAT:OPER:INST:ENAB'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def opr_inst_ptr_reg(self, reg_value=None):
        query = 'STAT:OPER:INST:PTR?'
        write = 'STAT:OPER:INST:PTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def opr_inst_ntr_reg(self, reg_value=None):
        query = 'STAT:OPER:INST:NTR?'
        write = 'STAT:OPER:INST:NTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    # @TODO: Register layout undocumented by R&S ....
    # #######################################################
    # # Operation event instrument channel summary register #
    # #######################################################

    def get_opr_inst_sum_cond_reg(self):
        query = 'STAT:OPER:INST:ISUM' + self._channel + ':COND?'
        return self._command.read(query)

    def get_opr_inst_sum_event_reg(self):
        query = 'STAT:OPER:INST:ISUM' + self._channel + ':EVEN?'
        return self._command.read(query)

    def opr_inst_sum_enable_reg(self, reg_value=None):
        query = 'STAT:OPER:INST:ISUM' + self._channel + ':ENAB?'
        write = 'STAT:OPER:INST:ISUM' + self._channel + ':ENAB'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def opr_inst_sum_ptr_reg(self, reg_value=None):
        query = 'STAT:OPER:INST:ISUM' + self._channel + ':PTR?'
        write = 'STAT:OPER:INST:ISUM' + self._channel + ':PTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def opr_inst_sum_ntr_reg(self, reg_value=None):
        query = 'STAT:OPER:INST:ISUM' + self._channel + ':NTR?'
        write = 'STAT:OPER:INST:ISUM' + self._channel + ':NTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    # ##########################################
    # # Questionable event instrument register #
    # ##########################################

    def get_ques_inst_cond_reg(self):
        query = 'STAT:QUES:INST:COND?'
        return self._command.read(query)

    def get_ques_inst_event_reg(self):
        query = 'STAT:QUES:INST:EVEN?'
        return self._command.read(query)

    def set_ques_inst_enable_reg(self, reg_value=None):
        query = 'STAT:QUES:INST:ENAB?'
        write = 'STAT:QUES:INST:ENAB'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def set_ques_inst_ptr_reg(self, reg_value=None):
        query = 'STAT:QUES:INST:PTR?'
        write = 'STAT:QUES:INST:PTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def set_ques_inst_ntr_reg(self, reg_value=None):
        query = 'STAT:QUES:INSTS:NTR?'
        write = 'STAT:QUES:INST:NTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    # ####################################################
    # # Questionable instrument channel summary register #
    # ####################################################

    def get_ques_inst_sum_cond_reg(self):
        query = 'STAT:QUES:INST:ISUM' + self._channel + ':COND?'
        return self._command.read(query)

    def get_ques_inst_sum_event_reg(self):
        query = 'STAT:QUES:INST:ISUM' + self._channel + ':EVEN?'
        return self._command.read(query)

    def ques_inst_sum_enable_reg(self, reg_value=None):
        query = 'STAT:QUES:INST:ISUM' + self._channel + ':ENAB?'
        write = 'STAT:QUES:INST:ISUM' + self._channel + ':ENAB'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def ques_inst_sum_ptr_reg(self, reg_value=None):
        query = 'STAT:QUES:INST:ISUM' + self._channel + ':PTR?'
        write = 'STAT:QUES:INST:ISUM' + self._channel + ':PTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)

    def ques_inst_sum_ntr_reg(self, reg_value=None):
        query = 'STAT:QUES:INST:ISUM' + self._channel + ':NTR?'
        write = 'STAT:QUES:INST:ISUM' + self._channel + ':NTR'
        return self._command.read_write(
            query, write, self._validate.register_16,
            reg_value)


