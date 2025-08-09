## 🔧 Configuration# EC2 Predictive Horizontal Auto Scaler

A machine learning-powered auto scaling solution for AWS EC2 instances that predicts optimal scaling decisions based on real-time metrics and workload patterns.

## 🎯 Project Overview

This project implements a predictive auto scaler that uses machine learning to make intelligent scaling decisions for AWS Auto Scaling Groups (ASG). Unlike traditional reactive auto scaling that responds to threshold breaches, this solution predicts scaling needs based on workload patterns and system metrics.

### Key Features

- **Predictive Scaling**: Uses ML models to predict scaling needs before resource constraints occur
- **Multi-metric Analysis**: Considers 9 different system and workload metrics
- **Conservative Scale-down**: Requires 5 minutes of consistent predictions before scaling down
- **Real-time Monitoring**: Collects metrics every 10 seconds for responsive scaling
- **Comprehensive Logging**: Detailed metrics collection and scaling decision logging

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│       ASG       │────│  Target EC2s    │
│   (ALB/NLB)     │    │   (asg-prod)    │    │  (Workload)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │ Monitors & Scales
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Monitor EC2   │────│  FastAPI ML     │
                       │ (Auto Scaler)   │    │   Predictor     │
                       └─────────────────┘    └─────────────────┘
```

## 📊 Metrics Used for Prediction

The ML model analyzes these 9 key metrics to make scaling decisions:

1. **cpu_usage** (45.5%) - Average CPU utilization across instances
2. **memory_usage** (60.2%) - Average memory utilization 
3. **network_traffic** (122.1 MB/s) - Combined network I/O traffic
4. **power_consumption** (35.6 watts) - Estimated total power consumption
5. **num_executed_instructions** (1.2e9) - Total instructions executed
6. **execution_time** (0.8s) - Estimated task execution time
7. **energy_efficiency** (0.92) - Performance-to-power ratio
8. **task_type** ("compute"/"network"/"io") - Workload classification
9. **task_priority** ("high"/"medium"/"low") - Priority classification

## 🚀 AWS Infrastructure Setup

### Prerequisites

- AWS Account with appropriate permissions
- Python 3.8+
- AWS CLI configured

### Step 1: AWS Infrastructure Setup

#### 1.1 Create Auto Scaling Group (ASG)

```bash
# Create launch template
aws ec2 create-launch-template \
  --launch-template-name "web-app-template" \
  --launch-template-data '{
    "ImageId": "ami-0c02fb55956c7d316",
    "InstanceType": "t3.small",
    "KeyName": "your-key-pair",
    "SecurityGroupIds": ["sg-xxxxxxxxx"],
    "UserData": "base64-encoded-startup-script"
  }'

# Create Auto Scaling Group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name "asg-prod" \
  --launch-template "LaunchTemplateName=web-app-template,Version=\$Latest" \
  --min-size 1 \
  --max-size 5 \
  --desired-capacity 2 \
  --target-group-arns "arn:aws:elasticloadbalancing:region:account:targetgroup/..." \
  --vpc-zone-identifier "subnet-xxxxx,subnet-yyyyy"
```

#### 1.2 Create Load Balancer

```bash
# Create Application Load Balancer
aws elbv2 create-load-balancer \
  --name "web-app-alb" \
  --subnets subnet-xxxxx subnet-yyyyy \
  --security-groups sg-xxxxxxxxx

# Create target group and register with ASG
aws elbv2 create-target-group \
  --name "web-app-targets" \
  --protocol HTTP \
  --port 80 \
  --vpc-id vpc-xxxxxxxxx \
  --health-check-path "/health"
```

#### 1.3 Create Monitor EC2 Instance

Launch a separate EC2 instance to run the auto scaler:

```bash
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.small \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxx \
  --iam-instance-profile Name="AutoScalerRole" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=AutoScaler-Monitor}]'
```

#### 1.4 Create IAM Role for Monitor EC2

Create an IAM role with the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "autoscaling:DescribeAutoScalingGroups",
        "autoscaling:SetDesiredCapacity",
        "cloudwatch:GetMetricStatistics",
        "ec2:DescribeInstances"
      ],
      "Resource": "*"
    }
  ]
}
```

```bash
# Create role
aws iam create-role \
  --role-name AutoScalerRole \
  --assume-role-policy-document file://trust-policy.json

# Attach policy
aws iam put-role-policy \
  --role-name AutoScalerRole \
  --policy-name AutoScalerPolicy \
  --policy-document file://permissions-policy.json

# Create instance profile
aws iam create-instance-profile --instance-profile-name AutoScalerRole
aws iam add-role-to-instance-profile \
  --instance-profile-name AutoScalerRole \
  --role-name AutoScalerRole
```

## 🔍 How the Scaler Works - Deep Dive

## 🔍 How the Scaler Works - Deep Dive

### System Architecture Flow

The predictive auto scaler operates through a continuous monitoring and decision-making loop:

1. **Metrics Collection** (every 10 seconds)
2. **Data Processing** (CloudWatch → Derived Metrics)
3. **ML Prediction** (FastAPI service)
4. **Scaling Decision** (Based on prediction + cooldown logic)
5. **AWS API Execution** (Modify ASG capacity)

### Detailed Component Breakdown

#### 1. Initialization and Configuration

The scaler initializes with ASG configuration and sets up monitoring parameters:

```python
def __init__(self, asg_name, region='eu-west-1', update_interval=10):
    self.asg_name = asg_name
    self.region = region
    self.update_interval = update_interval
    
    # AWS clients initialization
    self.cloudwatch = boto3.client('cloudwatch', region_name=region)
    self.autoscaling = boto3.client('autoscaling', region_name=region)
    self.ec2 = boto3.client('ec2', region_name=region)
    
    # Critical buffers for decision making
    self.metrics_buffer = deque(maxlen=1000)  # Rolling window of metrics
    self.scale_down_predictions = deque(maxlen=30)  # 5 minutes of predictions
    
    # Cooldown mechanism
    self.last_scaling_action = None
    self.last_scaling_time = None
    self.scaling_cooldown = 300  # 5 minutes
    
    # Get ASG limits from AWS
    self.asg_config = self.get_asg_configuration()
    self.min_capacity = self.asg_config['MinSize']
    self.max_capacity = self.asg_config['MaxSize']
```

#### 2. Metrics Collection Pipeline

The core metrics collection happens in `collect_metrics()`:

```python
def collect_metrics(self):
    # Step 1: Get all running instances in ASG
    instance_ids = self.get_asg_instances()
    if not instance_ids:
        return None
    
    # Step 2: Collect CloudWatch metrics from each instance
    all_metrics = []
    for instance_id in instance_ids:
        instance_metrics = self.fetch_cloudwatch_metrics(instance_id)
        if instance_metrics:
            all_metrics.append(instance_metrics)
    
    # Step 3: Combine metrics across all instances
    combined_cloudwatch = self.combine_instance_metrics(all_metrics)
    
    # Step 4: Transform into ML model features
    derived_metrics = self.calculate_derived_metrics(combined_cloudwatch, len(instance_ids))
    
    # Step 5: Make scaling decision
    self.handle_predictive_scaling(derived_metrics)
```

#### 3. CloudWatch Metrics Fetching

Each instance provides raw CloudWatch metrics:

```python
def fetch_cloudwatch_metrics(self, instance_id):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=3)  # 3-minute window
    
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
        response = self.cloudwatch.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=60,  # 1-minute granularity
            Statistics=['Average', 'Maximum']
        )
        
        if response['Datapoints']:
            latest = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])[-1]
            metrics_data[key] = {
                'value': round(latest['Average'], 2),
                'max_value': round(latest['Maximum'], 2),
                'timestamp': latest['Timestamp'].isoformat()
            }
```

#### 4. Multi-Instance Metrics Aggregation

Raw metrics from multiple instances are combined using statistical aggregation:

```python
def combine_instance_metrics(self, all_metrics):
    combined = {}
    metric_keys = ['cpu_utilization', 'network_in', 'network_out', 'disk_read_bytes', 
                  'disk_write_bytes', 'network_packets_in', 'network_packets_out']
    
    for key in metric_keys:
        values = []
        max_values = []
        
        # Collect values from all instances
        for instance_metrics in all_metrics:
            if key in instance_metrics and 'value' in instance_metrics[key]:
                values.append(instance_metrics[key]['value'])
                max_values.append(instance_metrics[key].get('max_value', 0))
        
        if values:
            combined[key] = {
                'value': round(np.mean(values), 2),      # Average across instances
                'max_value': round(np.max(max_values), 2), # Peak value observed
                'min_value': round(np.min(values), 2),     # Minimum observed
                'instance_count': len(values)
            }
    
    return combined
```

#### 5. Feature Engineering for ML Model

The system transforms raw CloudWatch metrics into the 9 features expected by the ML model:

```python
def calculate_derived_metrics(self, cloudwatch_data, instance_count):
    # Extract raw values
    cpu_raw = cloudwatch_data.get('cpu_utilization', {}).get('value', 0)
    network_in = cloudwatch_data.get('network_in', {}).get('value', 0)
    network_out = cloudwatch_data.get('network_out', {}).get('value', 0)
    memory_raw = cloudwatch_data.get('memory_utilization', {}).get('value', 0)
    
    # Feature 1: CPU Usage (%)
    cpu_usage = round(cpu_raw, 1)
    
    # Feature 2: Memory Usage (%) - Estimated if not available
    if memory_raw == 0:
        # Correlation-based estimation with noise
        memory_usage = round(min(cpu_raw * 1.2 + np.random.normal(0, 5), 95), 1)
    else:
        memory_usage = round(memory_raw, 1)
    
    # Feature 3: Network Traffic (MB/s) - Combined I/O across all instances
    network_traffic = round((network_in + network_out) * instance_count / (1024 * 1024), 1)
    
    # Feature 4: Power Consumption (watts) - Physics-based estimation
    base_power_per_instance = 20  # Base power for t3.small
    cpu_factor = cpu_usage / 100
    total_power = round(instance_count * (base_power_per_instance + 
                       (base_power_per_instance * 0.6 * cpu_factor)), 1)
    
    # Feature 5: Number of Executed Instructions - CPU performance model
    cpu_frequency = 2400  # MHz for typical instance
    instructions_per_cycle = 2.5  # IPC estimate
    utilization = cpu_usage / 100
    total_instructions = int(instance_count * utilization * cpu_frequency * 1e6 * instructions_per_cycle)
    
    # Feature 6: Execution Time - Workload complexity estimation
    workload_complexity = (cpu_usage + memory_usage + min(network_traffic * 10, 50)) / 3
    execution_time = round((0.3 + (workload_complexity / 100) * 2) / max(instance_count * 0.8, 1), 2)
    
    # Feature 7: Energy Efficiency - Performance per watt
    performance_metric = (cpu_usage + memory_usage) / 2 * instance_count
    energy_efficiency = round(performance_metric / max(total_power, 1), 4)
    
    # Feature 8: Task Type Classification
    task_type = self.classify_task_type(cpu_usage, memory_usage, network_traffic)
    
    # Feature 9: Task Priority Classification
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
```

#### 6. Workload Classification Logic

The system includes intelligent workload classification:

```python
def classify_task_type(self, cpu, memory, network):
    """Classify workload based on dominant resource usage"""
    if cpu >= max(network, memory):
        return "compute"  # CPU-bound workload
    elif network >= max(cpu, memory):
        return "network"  # Network-intensive workload
    else:
        return "io"      # Memory/IO-bound workload

def classify_task_priority(self, cpu, memory, network, instance_count):
    """Determine task priority based on resource intensity and capacity"""
    resource_intensity = max(cpu, memory, min(network * 2, 100))
    
    # Priority logic based on current capacity and resource usage
    if instance_count <= self.min_capacity and resource_intensity > 60:
        return "high"    # Low capacity + high usage = urgent
    elif instance_count >= self.max_capacity and resource_intensity < 30:
        return "low"     # Max capacity + low usage = can scale down
    elif resource_intensity > 80:
        return "high"    # Always high priority if resources stressed
    elif resource_intensity > 50:
        return "medium"
    else:
        return "low"
```

#### 7. ML Prediction Service Integration

The scaler communicates with the FastAPI service for predictions:

```python
def make_prediction(self, metrics):
    # Prepare feature vector for ML model
    features = [
        metrics['cpu_usage'],           # Feature 1
        metrics['memory_usage'],        # Feature 2 
        metrics['network_traffic'],     # Feature 3
        metrics['power_consumption'],   # Feature 4
        metrics['num_executed_instructions'], # Feature 5
        metrics['execution_time'],      # Feature 6
        metrics['energy_efficiency'],   # Feature 7
        metrics['task_type'],          # Feature 8 (categorical)
        metrics['task_priority']       # Feature 9 (categorical)
    ]
    
    payload = {"features": features}
    
    # API call with timeout
    response = requests.post(
        self.prediction_api_url,  # http://localhost:8000/predict
        json=payload,
        timeout=6
    )
    
    if response.status_code == 200:
        result = response.json()
        # Returns: {"prediction": 0/1/2, "predicted_class": "Low/Medium/High", "confidence": "..."}
        return result
```

#### 8. FastAPI ML Service Processing

The prediction service processes features through multiple stages:

```python
def preprocess_input_data(input_features):
    # Convert to DataFrame for processing
    input_df = pd.DataFrame([input_features], columns=[
        'cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption',
        'num_executed_instructions', 'execution_time', 'energy_efficiency', 
        'task_type', 'task_priority'
    ])
    
    # Add status column (always 'completed' for live data)
    input_df['task_status'] = 'completed'
    
    # Feature engineering - create interaction features
    input_df['cpu_mem_product'] = input_df['cpu_usage'] * input_df['memory_usage']
    input_df['cpu_network_ratio'] = input_df['cpu_usage'] / (input_df['network_traffic'] + 1e-5)
    input_df['power_per_cpu'] = input_df['power_consumption'] / (input_df['cpu_usage'] + 1e-5)
    input_df['execution_per_instruction'] = input_df['execution_time'] / (input_df['num_executed_instructions'] + 1)
    
    # Apply feature scaling to numerical columns
    scaling_cols = ['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption',
                   'num_executed_instructions', 'execution_time', 'energy_efficiency',
                   'cpu_mem_product', 'cpu_network_ratio', 'power_per_cpu', 'execution_per_instruction']
    input_df[scaling_cols] = scaler.transform(input_df[scaling_cols])
    
    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['task_type', 'task_priority', 'task_status'], drop_first=True)
    
    # Remove energy_efficiency if it exists (model doesn't use it)
    if 'energy_efficiency' in input_df.columns:
        input_df = input_df.drop('energy_efficiency', axis=1)
    
    # Ensure all expected features are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Return in correct feature order
    return input_df[feature_columns]
```

#### 9. Scaling Decision Logic

Based on the ML prediction, the system makes scaling decisions:

```python
def handle_predictive_scaling(self, metrics):
    prediction_result = self.make_prediction(metrics)
    
    if prediction_result:
        prediction = prediction_result.get('prediction')
        predicted_class = prediction_result.get('predicted_class', 'Unknown')
        confidence = prediction_result.get('confidence', 0)
        
        logger.info(f"PREDICTION: {prediction} ({predicted_class}) - Confidence: {confidence}")
        
        # Decision tree based on prediction
        if prediction == 0:      # Low class - need more resources
            self.handle_scale_up()
        elif prediction == 1:    # Medium class - can reduce resources  
            self.handle_scale_down()
        # prediction == 2 (High class) - maintain current capacity
```

#### 10. Scale Up Implementation

Scale up logic is immediate but respects cooldown and capacity limits:

```python
def handle_scale_up(self):
    current_config = self.get_asg_configuration()
    current_capacity = current_config['DesiredCapacity']
    
    # Check capacity limits
    if current_capacity < self.max_capacity:
        # Check cooldown period
        if self.can_perform_scaling():
            new_capacity = min(current_capacity + 1, self.max_capacity)
            logger.info(f"SCALING UP: {current_capacity} -> {new_capacity} instances")
            self.modify_asg_capacity(new_capacity)
        else:
            logger.info(f"Scale up needed but in cooldown period")
    else:
        logger.info(f"Already at maximum capacity of ({current_capacity} instances)")
```

#### 11. Scale Down Implementation (Conservative Approach)

Scale down requires 5 minutes of consistent predictions:

```python
def handle_scale_down(self):
    # Add current prediction to buffer
    self.scale_down_predictions.append(1)
    
    current_config = self.get_asg_configuration()
    current_capacity = current_config['DesiredCapacity']
    
    if current_capacity > self.min_capacity:
        # Require 30 consecutive predictions (5 minutes at 10-second intervals)
        if len(self.scale_down_predictions) >= 30:
            # All predictions must be scale-down (value = 1)
            if all(pred == 1 for pred in self.scale_down_predictions):
                if self.can_perform_scaling():
                    new_capacity = max(current_capacity - 1, self.min_capacity)
                    logger.info(f"SCALING DOWN: {current_capacity} -> {new_capacity} instances (5min consistent prediction)")
                    self.modify_asg_capacity(new_capacity)
                    self.scale_down_predictions.clear()  # Reset buffer after scaling
                else:
                    logger.info(f"Scale down needed but in cooldown period")
            else:
                logger.info(f"Mixed predictions - not scaling down")
```

#### 12. Cooldown Mechanism

Prevents rapid scaling oscillations:

```python
def can_perform_scaling(self):
    if self.last_scaling_time is None:
        return True  # First scaling action
    
    # Calculate time since last scaling action
    time_since_last_scaling = (datetime.utcnow() - self.last_scaling_time).total_seconds()
    return time_since_last_scaling >= self.scaling_cooldown  # 300 seconds = 5 minutes
```

#### 13. AWS API Execution

The actual scaling is performed through AWS Auto Scaling API:

```python
def modify_asg_capacity(self, new_desired_capacity):
    logger.info(f"Updating ASG desired capacity to {new_desired_capacity}...")
    
    # Update ASG desired capacity
    self.autoscaling.set_desired_capacity(
        AutoScalingGroupName=self.asg_name,
        DesiredCapacity=new_desired_capacity,
        HonorCooldown=False  # We manage our own cooldown
    )
    
    # Record scaling action
    self.last_scaling_action = f"capacity_{new_desired_capacity}"
    self.last_scaling_time = datetime.utcnow()
    
    # Wait for AWS to process the change
    time.sleep(30)
    
    # Verify the scaling action
    updated_config = self.get_asg_configuration()
    logger.info(f"ASG Status - Desired: {updated_config['DesiredCapacity']}, "
               f"Min: {updated_config['MinSize']}, Max: {updated_config['MaxSize']}, "
               f"Running: {len([i for i in updated_config['Instances'] if i['LifecycleState'] == 'InService'])}")
```

#### 14. Continuous Monitoring Loop

The entire process runs in a continuous loop:

```python
def monitoring_loop(self):
    while self.monitoring_active:
        start_time = time.time()
        
        # Execute one complete monitoring cycle
        self.collect_metrics()  # This triggers the entire pipeline
        
        # Maintain consistent timing
        elapsed = time.time() - start_time
        sleep_time = max(0, self.update_interval - elapsed)  # 10 seconds
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            logger.warning(f"Collection took {elapsed:.1f}s, longer than interval {self.update_interval}s")
```

### Key Design Decisions

1. **Conservative Scale-Down**: Requires 5 minutes of consistent predictions to prevent premature downsizing
2. **Immediate Scale-Up**: Responds quickly to resource pressure to maintain performance
3. **Cooldown Period**: 5 minute buffer between scaling actions to prevent oscillation
4. **Multi-Instance Aggregation**: Averages metrics across all ASG instances for holistic view
5. **Feature Engineering**: Transforms raw CloudWatch metrics into meaningful ML features
6. **Workload Classification**: Understands different types of workloads (compute/network/io)
7. **Priority Assessment**: Considers current capacity when determining urgency

### Scaling Parameters

```python
class EC2PredictiveHorizontalAutoScaler:
    def __init__(self, asg_name, region='eu-west-1', update_interval=10):
        # Scaling behavior
        self.scaling_cooldown = 300  # 5 minutes between scaling actions
        self.scale_down_predictions = deque(maxlen=30)  # Require 30 consistent predictions (5 min)
```

### ML Model Configuration

The FastAPI service (`app.py`) processes input features and returns predictions:

```python
# Prediction classes
class_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

# Feature engineering
def preprocess_input_data(input_features):
    # Creates derived features like cpu_mem_product, cpu_network_ratio, etc.
    # Applies scaling and one-hot encoding
```

## 📈 Monitoring and Logging

### Log Files

- **Application Logs**: `./output/ec2_monitor.log`
- **Metrics Data**: `./output/ec2_metrics_data.json`

### Key Log Messages

```
ASG METRICS (2 instances): {"cpu_usage": 45.5, "memory_usage": 60.2, ...}
PREDICTION: 0 (Low) - Confidence: Model prediction successful
SCALING UP: 2 -> 3 instances
SCALING DOWN: 3 -> 2 instances (5min consistent prediction)
```

### Monitoring Metrics

The scaler collects and analyzes:
- CloudWatch metrics from all ASG instances
- Derived performance metrics
- Scaling decisions and timing
- Prediction accuracy and confidence

## 🔒 Security Considerations

1. **IAM Permissions**: Use least-privilege access for the monitor EC2 role
2. **Network Security**: Restrict security group access to necessary ports only
3. **API Security**: The FastAPI service runs on localhost:8000 by default
4. **Logging**: Sensitive information should not be logged

## 🚨 Scaling Behavior

### Scale Up Logic
- **Trigger**: ML model predicts class 0 (High resource need)
- **Action**: Immediate scale up (if not in cooldown)
- **Increment**: +1 instance per scaling event

### Scale Down Logic
- **Trigger**: ML model predicts class 1 (Low resource need)
- **Condition**: Requires 30 consecutive predictions (5 minutes)
- **Action**: Scale down by 1 instance
- **Safety**: Never scales below minimum capacity

### Cooldown Period
- **Duration**: 300 seconds (5 minutes)
- **Purpose**: Prevents thrashing between scaling actions
- **Behavior**: Scaling actions are logged but not executed during cooldown

## 🔄 How It Works

1. **Metrics Collection**: Every 10 seconds, collect CloudWatch metrics from all ASG instances
2. **Feature Engineering**: Transform raw metrics into ML model features
3. **Prediction**: Send features to FastAPI service for ML prediction
4. **Decision Making**: Based on prediction, decide to scale up, down, or maintain
5. **Action Execution**: Execute scaling action via AWS Auto Scaling API
6. **Monitoring**: Log all actions and continue monitoring

## 🛠️ Troubleshooting

### Common Issues

1. **AWS Permissions**: Ensure the monitor EC2 has the correct IAM role attached
2. **API Connection**: Verify the FastAPI service is running on port 8000
3. **ASG Not Found**: Check ASG name and region configuration
4. **CloudWatch Metrics**: Ensure detailed monitoring is enabled on EC2 instances

### Debug Commands

```bash
# Check service status
sudo systemctl status ml-predictor auto-scaler

# View real-time logs
tail -f ./output/ec2_monitor.log

# Test API directly
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [45.5, 60.2, 122.1, 35.6, 1200000000, 0.8, 0.92, "compute", "high"]}'
```

## 📊 Performance Metrics

The system provides detailed performance tracking:
- Prediction accuracy and response times
- Scaling action frequency and effectiveness  
- Resource utilization trends
- Cost optimization metrics

## 🔮 Future Enhancements

- Support for multiple ASGs
- Custom metric weights and thresholds
- Integration with AWS Systems Manager for configuration
- Cost-aware scaling decisions
- A/B testing framework for model improvements

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review application logs
- Open an issue in the repository