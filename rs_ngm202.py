# -*- coding: utf-8 -*-

import pyvisa
import numpy as np

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
        self.com = Common(self._bus)
        self.disp = Display(self._bus)
        self.ch1 = Channel(self._bus, '1')
        if global_input_values['ch2']:
            self.ch2 = Channel(self._bus, '2')

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


class Common:
    def __init__(self, bus):
        self._bus = bus
        self._validate = ValidateRegister()
        self._command = Command(self._bus)

    # Clears event registers and errors
    def cls(self):
        write = "*CLS"
        self._command.write(write)

    # Read standard event enable register (no param)
    # Write with param
    def ese(self, reg_value=None):
        query = '*ESE?'
        write = '*ESE'
        return self._command.read_write(
            query, write, self._validate.register_8,
            reg_value)

    # Read and clear standard event enable register
    def esr(self):
        query = "*ESR?"
        return self._command.read(query)

    # Read instrument identification
    def idn(self):
        query = "*IDN?"
        return self._command.read(query)

    # Set the operation complete bit in the standard event register or queue
    # (param=1) places into output queue when operation complete
    def opc(self, reg_value=None):
        query = '*OPC?'
        write = '*OPC'
        return self._command.read_write(
            query, write, None, reg_value)

    # Returns the power supply to the saved setup (0...9)
    def rcl(self, preset_value=None):
        query = '*RCL?'
        write = '*RCL'
        return self._command.read_write(
            query, write, self._validate.preset,
            preset_value)

    # Returns the power supply to the *RST default conditions
    def rst(self):
        write = "*RST"
        self._command.write(write)
        self.cls()

    # Saves the present setup (1..9)
    def sav(self, preset_value=None):
        query = '*SAV?'
        write = '*SAV'
        return self._command.read_write(
            query, write, self._validate.preset,
            preset_value)

    # Programs the service request enable register
    def sre(self, reg_value=None):
        query = '*SRE?'
        write = '*SRE'
        return self._command.read_write(
            query, write, self._validate.register_8,
            reg_value)

    # Reads the status byte register
    def stb(self):
        query = "*STB?"
        return self._command.read(query)

    # command to trigger
    def trg(self):
        write = "*TRG"
        self._command.write(write)

    # Waits until all previous commands are executed
    def wait(self):
        write = "*WAI"
        self._command.write(write)

    # Perform self-tests
    def tst(self):
        query = "*TST"
        return self._command.read(query)


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
        self.values = {
            'device': global_input_values,
            'settings': self._chan}

        # Channel objects
        #self.com = Common(self._bus)
        self.arb = Arbitrary(self._bus, self._channel)
        self.flog = FastLog(self._bus, self._channel)
        #self.status = Status(self._bus, self._channel)
        self.meas = Measure(self._bus, self._channel)

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
    # Valid Input: 'auto', 5, 20, 0.1, 0.01
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
        write = 'DISP:TEXT ' + str(message)
        self._command.write(write)

    def clear_message(self):
        write = 'DISP:CLE'
        self._command.write(write)


# @TODO: 'slow' logging
class Log:
    pass


class FastLog:
    def __init__(self, bus, channel):
        self._bus = bus
        self._validate = ValidateLog()
        self._channel = channel
        self._command = Command(self._bus, self._channel)
        self._flog_enable = False if self.get_enable() == '0' else True
        self._flog = {}
        self._flog = {
            'enable': self._flog_enable,
            'sample_rate': self.sample_rate(),
            'sample_time': '0'}
        self.values = {
            'device': global_input_values,
            'settings': self._flog}
        self.flog_data = {}

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

    def target_scpi(self):
        write = ':FLOG:TARG SCPI'
        self._command.write(write)

    def sample_rate(self, set_sample_rate=None):
        query = ':FLOG:SRAT?'
        write = ':FLOG:SRAT'
        return self._command.read_write(
            query, write, self._validate.sample_rates,
            set_sample_rate, self._flog, 'sample_rate')

    def initialize(self):
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
    def start(self, sample_rate='S15', num_samples=1):
        counter = num_samples
        inst_channel_bit = 2 if self._channel == '1' else 4
        stb_opr_status_bit = 128
        flog_data_rdy_bit = 4096
        opr_event_bit = 8192
        self.initialize()
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
                                self.flog_data['voltage'] = data[0:-2:2]
                                self.flog_data['current'] = data[1:-1:2]
                            else:
                                self.flog_data['voltage'] = np.append(
                                        self.flog_data['voltage'], data[0:-2:2], axis=0)
                                self.flog_data['current'] = np.append(
                                        self.flog_data['current'], data[1:-1:2], axis=0)
                            counter -= 1
                            if counter == 0:
                                self.disable()
        self.flog_data['power'] = self.flog_data['voltage'] * self.flog_data['current']
        self._flog['sample_time'] = np.single(
                (float(num_samples) * 0.5) / self.flog_data['voltage'].size)
        self.flog_data['seconds'] = np.arange(
                0, self._flog['sample_time'] * self.flog_data['voltage'].size,
                self._flog['sample_time'])


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


# @TODO not implemented
class Battery:
    def __init__(self, bus):
        self._bus = bus
    pass

# Usage Example:
# dev.ch1.arb.clear()
# dev.ch1.arb.add.point(5, 1, 0.1, 1)  ideally run in a loop
# .
# .
# dev.ch1.arb.add_point(...)
# dev.ch1.arb.edit_point(1024,....)         opps point 1024 is wrong (not necessary step)
# dev.ch1.arb.build()                       transfer all arb points to device
# dev.ch1.arb.repetitions(2)                repeat arb 2x
# dev.ch1.arb.end_behavior('off')           turn off at end of sequence
# dev.ch1.arb.activate_on_channel()         activate created arb for channel 1
#      or
# dev.ch1.arb.transfer_to_other_channel()   activate created arb for channel 2
# dev.ch1.arb.enable()                      enable arb on channel
# dev.ch1.on()

# @TODO implement file usage, trigger
# @TODO bug in repetitions, needs more investigation
# Setting repetition works.  Returning the value you set works.
# The device just seems to refuse to use the set number.
# Repetitions can be set manually on programmed arb (on front panel).
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
                     'points': self._arb_count}
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
        val = self._validate.voltage(voltage)
        arb_data['voltage'] = val
        val = self._validate.current(current)
        arb_data['current'] = val
        val = self._validate.dwell_time(dwell_time)
        arb_data['dwell_time'] = val
        val = self._validate.interpolation(interpolation)
        arb_data['interpolation'] = val
        self._arb_count += 1
        if self._arb_count <= 4096:
            self.arb_list[str(self._arb_count)] = arb_data
            self._arb['points'] = self._arb_count
        else:
            print('Maximum arb points reached!')

    def edit_point(self, point, voltage, current, dwell_time, interpolation):
        for x in self.arb_list.keys():
            if str(point) == x:
                arb_data = {}
                val = self._validate.voltage(voltage)
                arb_data['voltage'] = val
                val = self._validate.current(current)
                arb_data['current'] = val
                val = self._validate.dwell_time(dwell_time)
                arb_data['dwell_time'] = val
                val = self._validate.interpolation(interpolation)
                arb_data['interpolation'] = val
                self.arb_list[x] = arb_data
                return None
        return print('Arb point not found!')

    def build(self):
        arb_data = ''
        for x in self.arb_list.keys():
            arb_data += self.arb_list[x]['voltage'] + ','
            arb_data += self.arb_list[x]['current'] + ','
            arb_data += self.arb_list[x]['dwell_time'] + ','
            arb_data += self.arb_list[x]['interpolation'] + ','
        write = 'ARB:DATA ' + arb_data[0:-1]
        self._command.write(write)

    def activate_on_channel(self):
        write = 'ARB:TRAN ' + self._channel
        self._command.write(write)

    def transfer_to_other_channel(self):
        if self._channel == '1':
            channel = '2'
        else:
            channel = '1'
        write = 'ARB:TRAN ' + channel
        self._command.write(write)

    def repetitions(self, set_num_repeats=None):
        query = 'ARB:REP?'
        write = 'ARB:REP'
        return self._command.read_write(
            query, write, self._validate.repetition,
            set_num_repeats)

    def end_behavior(self, set_end_behavior=None):
        query = 'ARB:BEH:END?'
        write = 'ARB:BEH:END'
        return self._command.read_write(
            query, write, self._validate.end_behavior,
            set_end_behavior)

# @TODO not implemented
class Protection:
    def __init__(self, bus, channel):
        self._bus = bus
        self._channel = channel
    pass


# @TODO not implemented
class DigitalIO:
    def __init__(self, bus, channel):
        self._bus = bus
        self._channel = channel
    pass


# @TODO not implemented
class Trigger:
    def __init__(self, bus):
        self._bus = bus
    pass


class Validate:

    def float_range(self):
        return lambda x, y: y[0] <= x <= y[1]

    def int_range(self):
        return lambda x, y: x in range(y[0], y[1] + 1)

    def find_element(self):
        return lambda x, y: x in y

    def error_text(self, warning_type, error_type):
        ansi_esc_seq = {'HEADER': '\033[95m',
                        'OKBLUE': '\033[94m',
                        'OKGREEN': '\033[92m',
                        'WARNING': '\033[93m',
                        'FAIL': '\033[91m',
                        'ENDC': '\033[0m',
                        'BOLD': '\033[1m',
                        'UNDERLINE': '\033[4m'
                        }
        return str(ansi_esc_seq[warning_type] + str(error_type) + ansi_esc_seq['ENDC'])

    def float_rng_and_str_tuples(self, validation_set, value, round_to):
        if isinstance(value, (float, int)):
            val = round(float(value), round_to)
            validator = self.float_range()
            if validator(val, validation_set[0]):
                return str(value)
            else:
                return ValueError('ValueError!\n'
                                  'Not in range:(float, int) {}\n'
                                  'or in set:(str) {}'.format(
                    validation_set[0],
                    validation_set[1]))
        elif isinstance(value, str):
            val = value.lower()
            validator = self.find_element()
            if validator(val, str(validation_set[1]).lower()):
                return val.upper()
            else:
                return ValueError('ValueError!\n'
                                  'Not in set:(str) {}\n'
                                  'or in range:(float, int) {}'.format(
                    validation_set[1],
                    validation_set[0]))
        else:
            return TypeError('TypeError!\n'
                             'Received type: {}\n'
                             'Valid types: {}, {}, {}'.format(
                type(value), int, float, str))

    def int_rng_and_str_tuples(self, validation_set, value):
        if isinstance(value, int):
            val = value
            validator = self.int_range()
            if validator(val, validation_set[0]):
                return str(value)
            else:
                return ValueError('ValueError!\n'
                                  'Not in range:(int) {}\n'
                                  'or in set:(str) {}'.format(
                    validation_set[0],
                    validation_set[1]))
        elif isinstance(value, str):
            val = value.lower()
            validator = self.find_element()
            if validator(val, str(validation_set[1]).lower()):
                return val.upper()
            else:
                return ValueError('ValueError!\n'
                                  'Not in set:(str) {}\n'
                                  'or in range:(int) {}'.format(
                    validation_set[1],
                    validation_set[0]))
        else:
            return TypeError('TypeError!\n'
                             'Received type: {}\n'
                             'Valid types: {}, {}'.format(
                type(value), int, str))

    def float_and_str_tuples(self, validation_set, value):
        if isinstance(value, (float, int)):
            validator = self.find_element()
            val = float(value)
            if validator(val, validation_set[0]):
                return str(value)
            else:
                return ValueError('ValueError!\n'
                                  'Not in set:(float, int) {}\n'
                                  'or in set:(str) {}'.format(
                    validation_set[0],
                    validation_set[1]))
        elif isinstance(value, str):
            val = value.lower()
            validator = self.find_element()
            if validator(val, str(validation_set[1]).lower()):
                return val.upper()
            else:
                return ValueError('ValueError!\n'
                                  'Not in set:(str) {}\n'
                                  'or in set:(float, str) {}'.format(
                    validation_set[1],
                    validation_set[0]))
        else:
            return TypeError('TypeError!\n'
                             'Received type: {}\n'
                             'Valid types: {}, {}, {}'.format(
                type(value), int, float, str))

    def int_and_str_tuples(self, validation_set, value):
        if isinstance(value, int):
            validator = self.find_element()
            val = float(value)
            if validator(val, validation_set[0]):
                return str(value)
            else:
                return ValueError('ValueError!\n'
                                  'Not in set:(int) {}\n'
                                  'or in set:(str) {}'.format(
                    validation_set[0],
                    validation_set[1]))
        elif isinstance(value, str):
            val = value.lower()
            validator = self.find_element()
            if validator(val, str(validation_set[1]).lower()):
                return val.upper()
            else:
                return ValueError('ValueError!\n'
                                  'Not in set:(str) {}\n'
                                  'or in set:(int) {}'.format(
                    validation_set[1],
                    validation_set[0]))
        else:
            return TypeError('TypeError!\n'
                             'Received type: {}\n'
                             'Valid types: {}, {}'.format(
                type(value), int, str))

    def float_rng_tuple(self, validation_set, value, round_to):
        if isinstance(value, (float, int)):
            val = round(float(value), round_to)
            validator = self.float_range()
            if validator(val, validation_set):
                return str(value)
            else:
                return ValueError('ValueError!\n'
                                  'Not in range:(float, int) {}'
                                  .format(validation_set))
        else:
            return TypeError('TypeError!\n'
                             'Received type: {}\n'
                             'Valid types: {}, {}'.format(
                type(value), int, float))

    def str_tuple(self, validation_set, value):
        if isinstance(value, str):
            val = value.lower()
            validator = self.find_element()
            if validator(val, str(validation_set).lower()):
                return val.upper()
            else:
                return ValueError('ValueError!\n'
                                  'Not in set:(str) {}'.format(
                    validation_set))
        else:
            return TypeError('TypeError!\n'
                             'Received type: {}\n'
                             'Valid types: {}'.format(
                type(value), str))

    def int_tuple(self, validation_set, value):
        if isinstance(value, int):
            val = value
            validator = self.find_element()
            if validator(val, validation_set):
                return str(val)
            else:
                return ValueError('ValueError!\n'
                                  'Not in set:(int) {}'.format(
                    validation_set))
        else:
            return TypeError('TypeError!\n'
                             'Received type: {}\n'
                             'Valid types: {}'.format(
                type(value), int))

    def int_rng_tuple(self, validation_set, value):
        if isinstance(value, int):
            val = value
            validator = self.int_range()
            if validator(val, validation_set):
                return str(val)
            else:
                return ValueError('ValueError!\n'
                                  'Not in range:(int) {}'.format(
                    validation_set))
        else:
            return TypeError('TypeError!\n'
                             'Received type: {}\n'
                             'Valid types: {}'.format(
                type(value), int))

    def halt_on_fail(self, value):
        if isinstance(value, (ValueError, TypeError)):
            raise value
        else:
            return value

class ValidateArbitrary(Validate):
    def __init__(self):
        Validate().__init__()

    def voltage(self, value):
        voltage_values = (0.0, 20.0)
        return self.halt_on_fail(self.float_rng_tuple(voltage_values, value, 3))

    def current(self, value):
        current_values = (0.001, 6.0)
        return self.halt_on_fail(self.float_rng_tuple(current_values, value, 4))

    def dwell_time(self, value):
        dwell_time_values = (0.001, 1728000.0)
        return self.halt_on_fail(self.float_rng_tuple(dwell_time_values, value, 3))

    def interpolation(self, value):
        interpolation_values = (0, 1)
        return self.halt_on_fail(self.int_tuple(interpolation_values, value))

    def channel(self, value):
        channel_values = ('1', '2')
        return self.str_tuple(channel_values, value)

    def repetition(self, value):
        repetition_values = (0, 65535)
        return self.int_rng_tuple(repetition_values, value)

    def end_behavior(self, value):
        repetition_values = ('off', 'hold')
        return self.str_tuple(repetition_values, value)


class ValidateChannel(Validate):
    def __init__(self):
        Validate().__init__()

    def voltage(self, value):
        voltage_values = (0.0, 20.0), ('min', 'max')
        return self.float_rng_and_str_tuples(voltage_values, value, 3)

    def current(self, value):
        current_values = (0.001, 6.0), ('min', 'max')
        return self.float_rng_and_str_tuples(current_values, value, 4)

    def impedance(self, value):
        impedance_values = (-0.05, 100.0), ('min', 'max')
        return self.float_rng_and_str_tuples(impedance_values, value, 3)

    def resistance(self, value):
        resistance_values = (0.0, 10000.0), ('min', 'max')
        return self.float_rng_and_str_tuples(resistance_values, value, 3)

    def current_range(self, value):
        current_range_values = (10.0, 1.0, 0.1, 0.01), ('auto', 'min', 'max')
        return self.float_and_str_tuples(current_range_values, value)

    def voltage_range(self, value):
        voltage_range_values = (20.0, 5.0), 'auto'
        return self.float_and_str_tuples(voltage_range_values, value)

    def mode(self, value):
        mode_values = ('auto', 'sour', 'source', 'sink')
        return self.str_tuple(mode_values, value)

    def on_off(self, value):
        on_off_values = (0, 1), ('on', 'off')
        return self.int_rng_and_str_tuples(on_off_values, value)

    def output_delay_duration(self, value):
        output_delay_duration_values = (0.001, 10.0), ('on', 'off')
        return self.float_rng_and_str_tuples(output_delay_duration_values, value, 3)

    def channel(self, value):
        channel_values = ('1', '2')
        return self.str_tuple(channel_values, value)

    def ramp_duration(self, value):
        ramp_duration_values = (0.01, 10.0), ('min', 'max', 'default')
        return self.float_rng_and_str_tuples(ramp_duration_values, value, 2)


class ValidateDisplay(Validate):
    def __init__(self):
        Validate().__init__()

    def brightness(self, value):
        channel_values = (0.0, 1.0), ('min', 'max')
        return self.float_rng_and_str_tuples(channel_values, value, 1)


class ValidateLog(Validate):
    def __init__(self):
        Validate().__init__()

    def sample_rates(self, value):
        sample_rates_values = ('S15', 'S30', 'S61',
                               'S122', 'S244', 'S488',
                               'S976', 'S1K953', 'S3K906',
                               'S7K812', 'S15K625', 'S31K25',
                               'S62K5', 'S125K', 'S250K',
                               'S500K'
                               )
        return self.str_tuple(sample_rates_values, value)


class ValidateRegister(Validate):
    def __init__(self):
        Validate().__init__()

    def register_8(self, value):
        register_values = (0, 128)
        return self.int_rng_tuple(register_values, value)

    def register_16(self, value):
        register_values = (0, 65535)
        return self.int_rng_tuple(register_values, value)

    def preset(self, value):
        preset_values = (0, 9)
        return self.int_rng_tuple(preset_values, value)


class ValidateTrigger(Validate):
    def __init__(self):
        Validate().__init__()

    def source (self, value):
        source_values = ('int', 'ext', 'min',
                         'max', 'def', 'default'
                         )
        return self.str_tuple(source_values, value)

    def level_low(self, value):
        level_low_values = (0, 0.5), ('auto', 'min', 'max',
                                      'def', 'default'
                                      )
        return self.float_rng_and_str_tuples(level_low_values, value, 3)

    def level_high(self, value):
        level_high_values = (0, 7.0), ('auto', 'min', 'max',
                                       'def', 'default'
                                       )
        return self.float_rng_and_str_tuples(level_high_values, value, 3)

    def level_dvm(self, value):
        level_dvm_values = (-5.999, 25.0), ('auto', 'min', 'max',
                                            'def', 'default'
                                            )
        return self.float_rng_and_str_tuples(level_dvm_values, value, 3)

    def count(self, value):
        count_values = (1, 100), ('min', 'max', 'def', 'default')
        return self.int_rng_and_str_tuples(count_values, value)

    def slope(self, value):
        slope_values = ('pos', 'neg', 'min',
                        'max', 'def', 'default'
                        )
        return self.str_tuple(slope_values, value)

    def offset(self, value):
        offset_values = (-5000, 50000), ('min', 'max', 'def', 'default')
        return self.int_rng_and_str_tuples(offset_values, value)

    def timeout(self, value):
        timeout_values = (0.001, 60), ('inf', 'min', 'max',
                                       'def', 'default')
        return self.float_rng_and_str_tuples(timeout_values, value, 3)


class Command(Validate):
    def __init__(self, bus, channel=None):
        Validate().__init__()
        self._bus = bus
        self._channel = channel

    # Used internally to select channel
    def set_channel(self):
        self._bus.write('INST:NSEL ' + self._channel)

    def read_write(self, query: str, write: str,
                   validator=None, value=None,
                   value_dict=None, value_key=None):
        if self._channel is not None:
            self.set_channel()
        if value is None:
            return self._bus.query(query)
        else:
            if validator is not None:
                val = validator(value)
                if isinstance(val, (ValueError, TypeError)):
                    print(self.error_text('WARNING', val))
                else:
                    write = write + ' ' + str(value)
                    self._bus.write(write)
                    if value_dict is not None:
                        value_dict[value_key] = self._bus.query(query)
                    return None

            else:
                write = write + ' ' + str(value)
                self._bus.write(write)
                if value_dict is not None:
                    value_dict[value_key] = self._bus.query(query)
                return None

    def read(self, query: str):
        if self._channel is not None:
            self.set_channel()
        return self._bus.query(query)

    def write(self, write: str, validator=None):
        if self._channel is not None:
            self.set_channel()
            if validator is None:
                self._bus.write(write)
            else:
                val = validator
                if isinstance(val, (ValueError, TypeError)):
                    print(self.error_text('WARNING', val))
                else:
                    self._bus.write(write)

    def write_value(self, write: str,
                   validator=None, value=None):
        if self._channel is not None:
            self.set_channel()
        if validator is not None:
            val = validator(value)
            if isinstance(val, (ValueError, TypeError)):
                print(self.error_text('WARNING', val))
            else:
                write = write + ' ' + str(value)
                self._bus.write(write)
                return None
        else:
            write = write + ' ' + str(value)
            self._bus.write(write)
            return None
