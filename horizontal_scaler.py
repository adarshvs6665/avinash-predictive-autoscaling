import boto3
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import signal
import sys
import os
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('./output/ec2_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EC2PredictiveHorizontalAutoScaler:
    def __init__(self, asg_name, region='eu-west-1', update_interval=10):
        self.asg_name = asg_name
        self.region = region
        self.update_interval = update_interval
        
        # Initialise AWS clients
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.autoscaling = boto3.client('autoscaling', region_name=region)
        self.ec2 = boto3.client('ec2', region_name=region)
        
        # Monitoring state
        self.monitoring_active = False
        
        # To keep last 1000 data only
        self.metrics_buffer = deque(maxlen=1000)
        
        # File for storing all the metrics
        self.metrics_file = './output/ec2_metrics_data.json'
        self.metrics_data = []
        
        # Prediction API endpoint
        self.prediction_api_url = 'http://localhost:8000/predict'
        
        # Keep last 30 predictions only
        self.scale_down_predictions = deque(maxlen=30)  
        self.last_scaling_action = None
        self.last_scaling_time = None
        self.scaling_cooldown = 300
        
        # Get ASG configuration
        self.asg_config = self.get_asg_configuration()
        self.min_capacity = self.asg_config['MinSize']
        self.max_capacity = self.asg_config['MaxSize']
        
        logger.info(f"Initialized EC2 predictive horizontal auto scaler for ASG: {asg_name}")
        logger.info(f"ASG capacity range: {self.min_capacity} - {self.max_capacity} instances")

    def get_asg_configuration(self):
        try:
            response = self.autoscaling.describe_auto_scaling_groups(
                AutoScalingGroupNames=[self.asg_name]
            )
            
            if response['AutoScalingGroups']:
                asg = response['AutoScalingGroups'][0]
                return {
                    'MinSize': asg['MinSize'],
                    'MaxSize': asg['MaxSize'],
                    'DesiredCapacity': asg['DesiredCapacity'],
                    'Instances': asg['Instances']
                }
            else:
                raise Exception(f"ASG {self.asg_name} not found")
                
        except Exception as e:
            logger.error(f"Error getting ASG configuration: {str(e)}")
            raise

    def get_asg_instances(self):
        try:
            asg_config = self.get_asg_configuration()
            instance_ids = [instance['InstanceId'] for instance in asg_config['Instances'] 
                          if instance['LifecycleState'] == 'InService']
            return instance_ids
        except Exception as e:
            logger.error(f"Error getting ASG instances: {str(e)}")
            return []

    def collect_metrics(self):
        try:
            instance_ids = self.get_asg_instances()
            
            if not instance_ids:
                logger.warning("No running instances found in ASG")
                return None
            
            logger.info(f"Collecting metrics from {len(instance_ids)} instances: {instance_ids}")
            
            # Get CloudWatch metrics for all instances in ASG
            all_metrics = []
            for instance_id in instance_ids:
                instance_metrics = self.fetch_cloudwatch_metrics(instance_id)
                if instance_metrics:
                    all_metrics.append(instance_metrics)
            
            if not all_metrics:
                logger.warning("No metrics collected from any instance")
                return None
            
            # Combine metrics from all instances inside the ASG
            combined_cloudwatch = self.combine_instance_metrics(all_metrics)

            # Derive required metrics for the ML model
            derived_metrics = self.calculate_derived_metrics(combined_cloudwatch, len(instance_ids))

            metrics_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'asg_name': self.asg_name,
                'instance_count': len(instance_ids),
                'instance_ids': instance_ids,
                'raw_cloudwatch': combined_cloudwatch,
                'metrics': derived_metrics
            }
            
            self.metrics_buffer.append(metrics_entry)
            self.save_metrics_to_file(metrics_entry)
            self.log_metrics(derived_metrics, len(instance_ids))
            
            # Make predictive scaling decision
            self.handle_predictive_scaling(derived_metrics)
            
            return metrics_entry
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            
            
            return None

    def combine_instance_metrics(self, all_metrics):
        combined = {}
        
        metric_keys = ['cpu_utilization', 'network_in', 'network_out', 'disk_read_bytes', 
                      'disk_write_bytes', 'network_packets_in', 'network_packets_out',
                      'memory_utilization', 'disk_space_utilization']
        
        for key in metric_keys:
            values = []
            max_values = []
            
            for instance_metrics in all_metrics:
                if key in instance_metrics and 'value' in instance_metrics[key]:
                    values.append(instance_metrics[key]['value'])
                    max_values.append(instance_metrics[key].get('max_value', 0))
            
            if values:
                combined[key] = {
                    'value': round(np.mean(values), 2),
                    'max_value': round(np.max(max_values), 2),
                    'min_value': round(np.min(values), 2),
                    'instance_count': len(values)
                }
            else:
                combined[key] = {'value': 0, 'max_value': 0, 'status': 'no_data'}
        
        return combined

    def fetch_cloudwatch_metrics(self, instance_id):
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=3)
        
        metrics_config = [
            ('CPUUtilization', 'AWS/EC2', 'cpu_utilization'),
            ('NetworkIn', 'AWS/EC2', 'network_in'),
            ('NetworkOut', 'AWS/EC2', 'network_out'),
            ('DiskReadBytes', 'AWS/EC2', 'disk_read_bytes'),
            ('DiskWriteBytes', 'AWS/EC2', 'disk_write_bytes'),
            ('NetworkPacketsIn', 'AWS/EC2', 'network_packets_in'),
            ('NetworkPacketsOut', 'AWS/EC2', 'network_packets_out'),
        ]
        
        metrics_data = {}
        
        for metric_name, namespace, key in metrics_config:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace=namespace,
                    MetricName=metric_name,
                    Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=60,
                    Statistics=['Average', 'Maximum']
                )
                
                if response['Datapoints']:
                    latest = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])[-1]
                    metrics_data[key] = {
                        'value': round(latest['Average'], 2),
                        'max_value': round(latest['Maximum'], 2),
                        'timestamp': latest['Timestamp'].isoformat()
                    }
                else:
                    metrics_data[key] = {'value': 0, 'max_value': 0, 'status': 'no_data'}
                    
            except Exception as e:
                logger.warning(f"Failed to collect {metric_name} for {instance_id}: {str(e)}")
                
                metrics_data[key] = {'value': 0, 'max_value': 0, 'error': str(e)}
        
        return metrics_data

    def calculate_derived_metrics(self, cloudwatch_data, instance_count):
        cpu_raw = cloudwatch_data.get('cpu_utilization', {}).get('value', 0)
        network_in = cloudwatch_data.get('network_in', {}).get('value', 0)
        network_out = cloudwatch_data.get('network_out', {}).get('value', 0)
        memory_raw = cloudwatch_data.get('memory_utilization', {}).get('value', 0)
        
        # 1. cpu_usage (%) - Average across all instances
        cpu_usage = round(cpu_raw, 1)
        
        # 2. memory_usage (%) - Average across all instances
        if memory_raw == 0:
            memory_usage = round(min(cpu_raw * 1.2 + np.random.normal(0, 5), 95), 1)
        else:
            memory_usage = round(memory_raw, 1)
        
        # 3. network_traffic (MB/s) - Total across all instances
        network_traffic = round((network_in + network_out) * instance_count / (1024 * 1024), 1)
        
        # 4. power_consumption (watts) - Total for all instances
        base_power_per_instance = 20  # Assuming t3.small equivalent
        cpu_factor = cpu_usage / 100
        total_power = round(instance_count * (base_power_per_instance + (base_power_per_instance * 0.6 * cpu_factor)), 1)
        
        # 5. num_executed_instructions - Total across all instances
        cpu_frequency = 2400  # MHz
        instructions_per_cycle = 2.5
        utilization = cpu_usage / 100
        total_instructions = int(instance_count * utilization * cpu_frequency * 1e6 * instructions_per_cycle)
        
        # 6. execution_time (estimated based on workload distribution)
        workload_complexity = (cpu_usage + memory_usage + min(network_traffic * 10, 50)) / 3
        execution_time = round((0.3 + (workload_complexity / 100) * 2) / max(instance_count * 0.8, 1), 2)
        
        # 7. energy_efficiency (performance to power ratio)
        performance_metric = (cpu_usage + memory_usage) / 2 * instance_count
        energy_efficiency = round(performance_metric / max(total_power, 1), 4)
        
        # 8. task_type (classification based on workload pattern)
        task_type = self.classify_task_type(cpu_usage, memory_usage, network_traffic)
        
        # 9. task_priority (classification based on resource intensity)
        task_priority = self.classify_task_priority(cpu_usage, memory_usage, network_traffic, instance_count)
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'network_traffic': network_traffic,
            'power_consumption': total_power,
            'num_executed_instructions': total_instructions,
            'execution_time': execution_time,
            'energy_efficiency': energy_efficiency,
            'task_type': task_type,
            'task_priority': task_priority,
            'instance_count': instance_count
        }

    def classify_task_type(self, cpu, memory, network):
        if cpu >= max(network, memory):
            return "compute"
        elif network >= max(cpu, memory):
            return "network"
        else:
            return "io"

    def classify_task_priority(self, cpu, memory, network, instance_count):
        resource_intensity = max(cpu, memory, min(network * 2, 100))
        
        if instance_count <= self.min_capacity and resource_intensity > 60:
            return "high"
        elif instance_count >= self.max_capacity and resource_intensity < 30:
            return "low"
        elif resource_intensity > 80:
            return "high"
        elif resource_intensity > 50:
            return "medium"
        else:
            return "low"

    def log_metrics(self, metrics, instance_count):
        logger.info(f"ASG METRICS ({instance_count} instances): {json.dumps(metrics)}")

    def handle_predictive_scaling(self, metrics):
        try:
            prediction_result = self.make_prediction(metrics)
            
            if prediction_result is not None:
                prediction = prediction_result.get('prediction')
                predicted_class = prediction_result.get('predicted_class', 'Unknown')
                confidence = prediction_result.get('confidence', 0)
                
                logger.info(f"PREDICTION: {prediction} ({predicted_class}) - Confidence: {confidence}")
                
                if prediction == 0:
                    self.handle_scale_up()
                elif prediction == 1:
                    self.handle_scale_down()
                    
        except Exception as e:
            logger.error(f"Error in predictive scaling: {str(e)}")

    def make_prediction(self, metrics):
        try:
            features = [
                metrics['cpu_usage'],
                metrics['memory_usage'], 
                metrics['network_traffic'],
                metrics['power_consumption'],
                metrics['num_executed_instructions'],
                metrics['execution_time'],
                metrics['energy_efficiency'],
                metrics['task_type'],
                metrics['task_priority']
            ]
            
            payload = {"features": features}
            
            logger.info(f"Calling prediction API with features: {features}")
            
            response = requests.post(
                self.prediction_api_url,
                json=payload,
                timeout=6
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Prediction API response: {result}")
                return result
            else:
                logger.error(f"Prediction API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call prediction API: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error in make_prediction: {str(e)}")
            return None

    def handle_scale_up(self):
        try:
            current_config = self.get_asg_configuration()
            current_capacity = current_config['DesiredCapacity']
            
            if current_capacity < self.max_capacity:
                if self.can_perform_scaling():
                    new_capacity = min(current_capacity + 1, self.max_capacity)
                    logger.info(f"SCALING UP: {current_capacity} -> {new_capacity} instances")
                    self.modify_asg_capacity(new_capacity)
                else:
                    logger.info(f"Scale up needed but in cooldown period")
            else:
                logger.info(f"Already at maximum capacity of ({current_capacity} instances)")
                
        except Exception as e:
            logger.error(f"Error in handle_scale_up: {str(e)}")

    def handle_scale_down(self):
        try:
            self.scale_down_predictions.append(1)
            
            current_config = self.get_asg_configuration()
            current_capacity = current_config['DesiredCapacity']
            
            if current_capacity > self.min_capacity:
                # Check if we have 5 minutes of consistent scale-down predictions
                if len(self.scale_down_predictions) >= 30:
                    if all(pred == 1 for pred in self.scale_down_predictions):
                        if self.can_perform_scaling():
                            new_capacity = max(current_capacity - 1, self.min_capacity)
                            logger.info(f"SCALING DOWN: {current_capacity} -> {new_capacity} instances (5min consistent prediction)")
                            self.modify_asg_capacity(new_capacity)
                            self.scale_down_predictions.clear()  # Reset buffer
                        else:
                            logger.info(f"Scale down needed but in cooldown period")
                    else:
                        logger.info(f"Mixed predictions - not scaling down")
            else:
                logger.info(f"Already at minimum capacity ({current_capacity} instances)")
                
        except Exception as e:
            logger.error(f"Error in handle_scale_down: {str(e)}")

    def can_perform_scaling(self):
        if self.last_scaling_time is None:
            return True
            
        time_since_last_scaling = (datetime.utcnow() - self.last_scaling_time).total_seconds()
        return time_since_last_scaling >= self.scaling_cooldown

    def modify_asg_capacity(self, new_desired_capacity):
        try:
            logger.info(f"Updating ASG desired capacity to {new_desired_capacity}...")
            
            self.autoscaling.set_desired_capacity(
                AutoScalingGroupName=self.asg_name,
                DesiredCapacity=new_desired_capacity,
                HonorCooldown=False
            )
            
            self.last_scaling_action = f"capacity_{new_desired_capacity}"
            self.last_scaling_time = datetime.utcnow()
            
            logger.info(f"Successfully updated ASG desired capacity to {new_desired_capacity}")
            
            time.sleep(30)
            updated_config = self.get_asg_configuration()
            logger.info(f"ASG Status - Desired: {updated_config['DesiredCapacity']}, "
                       f"Min: {updated_config['MinSize']}, Max: {updated_config['MaxSize']}, "
                       f"Running: {len([i for i in updated_config['Instances'] if i['LifecycleState'] == 'InService'])}")
            
        except Exception as e:
            logger.error(f"Failed to modify ASG capacity: {str(e)}")
            raise

    def save_metrics_to_file(self, metrics_entry):
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    self.metrics_data = json.load(f)
            
            self.metrics_data.append(metrics_entry)
            
            # Keep only last 10000 entries
            if len(self.metrics_data) > 10000:
                self.metrics_data = self.metrics_data[-10000:]
            
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics to file: {str(e)}")

    def monitoring_loop(self):
        logger.info("Starting metrics collection loop for ASG")
        
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                self.collect_metrics()
                
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Collection took {elapsed:.1f}s, longer than interval {self.update_interval}s")
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.update_interval)
        
        logger.info("Monitoring loop stopped")

    def start_monitoring(self):
        if not self.monitoring_active:
            self.monitoring_active = True
            monitoring_thread = threading.Thread(
                target=self.monitoring_loop,
                name="ASGMetricsLogger",
                daemon=True
            )
            monitoring_thread.start()
            logger.info("ASG metrics logging started")

    def stop_monitoring(self):
        self.monitoring_active = False
        logger.info("ASG metrics logging stopped")

    def run(self):
        """Run the EC2 predictive horizontal auto scaler"""
        print("\n" + "="*70)
        print("EC2 Predictive Horizontal Auto Scaler")
        print("="*70)
        print(f"Monitoring ASG: {self.asg_name}")
        print(f"Region: {self.region}")
        print(f"Metrics Collection Interval: {self.update_interval} seconds")
        print(f"Capacity Range: {self.min_capacity} - {self.max_capacity} instances")
        print("="*70)
        
        self.start_monitoring()
        
        try:
            while self.monitoring_active:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping predictive horizontal auto scaler...")
            self.stop_monitoring()

def signal_handler(sig, frame):
    print('\nStopping...')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    asg_name = 'asg-prod'
    region = 'eu-west-1'
    update_interval = 10
    
    scaler_instance = EC2PredictiveHorizontalAutoScaler(
        asg_name=asg_name,
        region=region,
        update_interval=update_interval
    )
    
    try:
        scaler_instance.run()
    except Exception as e:
        logger.error(f"Failed to start horizontal auto scaler: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
