import boto3
import json
import statistics  # Import for median calculation
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

# Initialize DynamoDB
dynamodb = boto3.resource("dynamodb", region_name="us-west-2")
table = dynamodb.Table("BenchmarkMetrics")


# Helper functions
def get_latest_run_id(streaming):
    response = table.scan(
        FilterExpression="#streaming = :streaming",
        ExpressionAttributeNames={"#streaming": "streaming"},
        ExpressionAttributeValues={":streaming": streaming},
    )
    items = response.get("Items", [])
    if not items:
        return {"error": "No data found"}
    latest_item = max(items, key=lambda x: x["timestamp"])
    return {"run_id": latest_item["run_id"]}

def scan_all_items(scan_kwargs):
    items = []
    backoff = 1  # starting backoff time in seconds
    while True:
        try:
            response = table.scan(**scan_kwargs)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ProvisionedThroughputExceededException":
                print("Throughput exceeded, backing off for", backoff, "seconds")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)  # Exponential backoff up to a maximum
                continue
            else:
                raise e
        else:
            backoff = 1  # Reset backoff if successful
        items.extend(response.get("Items", []))
        if "LastEvaluatedKey" in response:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        else:
            break
    return items

def get_latest_vllm():
    """
    Retrieves the latest item with provider_name 'vLLM'.
    """
    scan_kwargs = {
        "FilterExpression":"#provider_name = :vllm",
        "ExpressionAttributeNames":{"#provider_name": "provider_name"},
        "ExpressionAttributeValues":{":vllm": "vLLM"},
    }
    items = scan_all_items(scan_kwargs)
    if not items:
        return {}
    latest_item = max(items, key=lambda x: x["timestamp"])
    return latest_item

def get_metrics(run_id, metricType=None):
    response = table.scan(
        FilterExpression="run_id = :run_id",
        ExpressionAttributeValues={":run_id": run_id},
    )
    items = response.get("Items", [])
    metrics_by_provider = {}

    for item in items:
        provider_name = item["provider_name"]
        model_name = item["model_name"]
        metrics = json.loads(item["metrics"])

        if provider_name not in metrics_by_provider:
            metrics_by_provider[provider_name] = {}

        if metricType:
            filtered_metrics = {metricType: metrics.get(metricType)} if metricType in metrics else {}
            metrics_by_provider[provider_name][model_name] = filtered_metrics
        else:
            metrics_by_provider[provider_name][model_name] = metrics

    return {"run_id": run_id, "metrics": metrics_by_provider}


def get_metrics_period(metricType, timeRange, streaming=True):
    time_ranges = {
        "week": timedelta(weeks=1),
        "month": timedelta(days=30),
        "three-month": timedelta(days=90),
    }

    if timeRange not in time_ranges:
        return {"error": f"Invalid timeRange. Valid options: {list(time_ranges.keys())}"}

    end_date = datetime.now()
    start_date = end_date - time_ranges[timeRange]
    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    scan_kwargs = {
            "FilterExpression": "#ts BETWEEN :start_date AND :end_date AND #streaming = :streaming AND #model_key = :common",
            "ExpressionAttributeNames": {
                "#ts": "timestamp",
                "#streaming": "streaming",
                "#model_key": "model_key"
            },
            "ExpressionAttributeValues": {
                ":start_date": start_date_str,
                ":end_date": end_date_str,
                ":streaming": streaming,
                ":common": "common"
            },
        }

    items = scan_all_items(scan_kwargs)
    aggregated_metrics = {}
    date_array = set()

    for item in items:
        provider_name = item["provider_name"]
        metrics = json.loads(item["metrics"])

        if metricType not in metrics:
            continue

        metric_data = metrics[metricType]
        latencies = list(map(float, metric_data.get("latencies", [])))
        if latencies:
            median_latency = statistics.median(latencies)  # Compute median instead of average
            formatted_date = datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%d-%m-%Y")
            date_array.add(formatted_date)

            if provider_name not in aggregated_metrics:
                aggregated_metrics[provider_name] = {}
            if formatted_date not in aggregated_metrics[provider_name]:
                aggregated_metrics[provider_name][formatted_date] = []
            aggregated_metrics[provider_name][formatted_date].append(median_latency)

    # Compute the median per date
    sorted_result = {
        provider: [{"date": date, "aggregated_metric": statistics.median(values)}  # Use median
                   for date, values in dates.items()]
        for provider, dates in aggregated_metrics.items()
    }

    sorted_date_array = sorted(date_array, key=lambda x: datetime.strptime(x, "%d-%m-%Y"), reverse=False)

    return {"metricType": metricType, "timeRange": timeRange, "aggregated_metrics": sorted_result, "date_array": sorted_date_array}


def get_metrics_by_date(metricType, date, streaming=True):
    if date == "latest":
        latest_id_response = get_latest_run_id(streaming)
        if "error" in latest_id_response:
            return {"error": "No latest run_id found."}
        run_id = latest_id_response["run_id"]
        result = get_metrics(run_id, metricType)
    else:
        try:
            start_date = datetime.strptime(date, "%d-%m-%Y")
            end_date = start_date + timedelta(days=1)
        except ValueError:
            return {"error": "Invalid date format. Use '12-12-2024'."}

        start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        scan_kwargs = {
            "FilterExpression": "#ts BETWEEN :start_date AND :end_date AND #streaming = :streaming AND #model_key = :common",
            "ExpressionAttributeNames": {
                "#ts": "timestamp",
                "#streaming": "streaming",
                "#model_key": "model_key"
            },
            "ExpressionAttributeValues": {
                ":start_date": start_date_str,
                ":end_date": end_date_str,
                ":streaming": streaming,
                ":common": "common"
            },
}
        items = scan_all_items(scan_kwargs)
        metrics_by_provider = {}
        for item in items:
            provider_name = item["provider_name"]
            model_name = item["model_name"]
            metrics = json.loads(item["metrics"])

            if provider_name not in metrics_by_provider:
                metrics_by_provider[provider_name] = {}

            if metricType:
                filtered_metrics = {metricType: metrics.get(metricType)} if metricType in metrics else {}
                metrics_by_provider[provider_name][model_name] = filtered_metrics
            else:
                metrics_by_provider[provider_name][model_name] = metrics

        sorted_metrics_by_provider = {
            provider: metrics_by_provider[provider]
            for provider in sorted(metrics_by_provider)
        }
        result = {"date": date, "metricType": metricType, "metrics": sorted_metrics_by_provider}

    # Append vLLM metrics to the result
    vllm_item = get_latest_vllm()
    if vllm_item and "vLLM" not in result["metrics"]:
        try:
            vllm_metrics = json.loads(vllm_item["metrics"])
            if metricType in vllm_metrics:
                # Insert vLLM metrics under a "vLLM" key, using the model_name as a sub-key.
                result["metrics"]["vLLM"] = {
                    vllm_item["model_name"]: {metricType: vllm_metrics[metricType]}
                }
        except Exception as e:
            print("Error processing vLLM metrics for date endpoint:", e)

    return result


# Lambda function handler
def lambda_handler(event, context):
    """Main Lambda handler function."""
    print("Received event:", json.dumps(event))  # Debugging

    path = event.get("rawPath", "/")
    params = event.get("queryStringParameters", {}) or {}

    if path == "/default":
        return {"statusCode": 200, "body": json.dumps({"status": "200", "message": "Welcome to the Benchmark Metrics API!"})}

    elif path == "/default/metrics/period":
        metricType = params.get("metricType")
        timeRange = params.get("timeRange")
        streaming = params.get("streaming", "true").lower() == "true"

        if not metricType or not timeRange:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing metricType or timeRange parameter"})}

        response = get_metrics_period(metricType, timeRange, streaming)
        return {"statusCode": 200, "body": json.dumps(response)}

    elif path == "/default/metrics/date":
        metricType = params.get("metricType")
        date = params.get("date")
        streaming = params.get("streaming", "true").lower() == "true"

        if not metricType or not date:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing metricType or date parameter lollllll"})}

        response = get_metrics_by_date(metricType, date, streaming)
        return {"statusCode": 200, "body": json.dumps(response)}
    elif path == "/default/metrics/vllm":
        vllm_item = get_latest_vllm()
        print(vllm_item)
        if not vllm_item:
            return {"statusCode": 404, "body": json.dumps({"error": "No vLLM metrics found"})}
        return {"statusCode": 200, "body": json.dumps(vllm_item)}

    else:
        return {"statusCode": 404, "body": json.dumps({"error": f"Invalid route: {path}"})}
