import holidays
from requests import post
from lib.executable import Executable
from configparser import ConfigParser
from datetime import datetime, timedelta, timezone
# from modules.influxdb_wrapper import Influx_wrapper

class Runner(Executable):

    def __init__(self, config: ConfigParser) -> None:
        super().__init__(config)

    def execute(self) -> None:
        time_now = datetime.now(timezone.utc).replace(tzinfo = None)

        # influx = Influx_wrapper(
        #     "https://hlx.adfomni.com/",
        #     "HLX",
        #     self.config[self.cls_name]['influx_token']
        # )

        differential_pressure=0.24
        # differential_pressure = influx.influx_query_last(
        #     "HLX",
        #     "Optimization",
        #     "Predictions",
        #     "optimized_differential_pressure",
        #     time_now - timedelta(days = 16),
        #     time_now
        # )
        # print (f"differential_pressure: {differential_pressure}")

        try:
            response = post(
                url = self.config['Post_office']['url'],
                json = {
                    'Header': {
                      'Version': 'HLX_optimization_schema_v1',
                      'RMQ_username': self.config['Post_office']['username'],
                      'RMQ_password': self.config['Post_office']['password'],
                      'RMQ_queue': self.config[self.cls_name]['rmq_queue'],
                      'Skip_version_check': True
                    },
                    'Body': {
                        'Bacnet_control_device':  {
                            'name': 'Local/Omni Control',
                            'address': '192.168.1.45',
                            'port': 47808,
                            'object_type': 'binaryValue:2500',
                            'property_value': 'presentValue'
                        },
                        'Bacnet_devices': [
                            {
                                'name': 'Differential Pressure Control',
                                'address': '192.168.1.3',
                                'port': 47808,
                                'setpoint': differential_pressure,
                                'object_type': 'analogValue:4',
                                'property_value': 'presentValue'
                            }
                        ]
                    }
                }
            )
            print(response)
        except Exception as error:
            print(f"Exception {error}")
            self.logger.log_error(f"Error sending optimization request to Omni post. | {error}")
        else:
            print(f"status: {response.status_code}")
            if response.status_code != 200:
                self.logger.log_error(f"Error sending optimization request to Omni post. | {response.status_code} | {response.text}")
            else:
                self.logger.log_info(f"Successfully sent optimization request request to Omni post.")


if __name__ == "__main__":
    config = ConfigParser()
    config.read('config.ini')

    Runner(config).execute()
