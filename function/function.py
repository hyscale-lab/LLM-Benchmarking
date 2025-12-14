import boto3
import json
import time
import statistics  # Import for median calculation
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

# Initialize DynamoDB
dynamodb = boto3.resource("dynamodb", region_name="us-west-2")
table = dynamodb.Table("BenchmarkMetrics")


# Helper functions
def add_input_type_filter(filter_exp, exp_names, exp_values, input_type):
    """
    Helper to append the input_type logic to scan kwargs.
    If inputType is 'static', it includes items where the column is MISSING.
    """
    if input_type == "static":
        # LOGIC: (input_type == 'static') OR (input_type does not exist)
        filter_exp += " AND ( #input_type = :input_type OR attribute_not_exists(#input_type) )"
    else:
        # LOGIC: input_type == 'trace' (strict match)
        filter_exp += " AND #input_type = :input_type"
    
    exp_names["#input_type"] = "input_type"
    exp_values[":input_type"] = input_type
    
    return filter_exp, exp_names, exp_values

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

def get_latest_vllm(streaming, input_type):
    """
    Retrieves the latest item with provider_name 'vLLM'.
    """
    filter_expression = "#provider_name = :vllm AND #streaming = :streaming"
    expression_names = {"#provider_name": "provider_name", "#streaming": "streaming"}
    expression_values = {":vllm": "vLLM", ":streaming": streaming}

    filter_expression, expression_names, expression_values = add_input_type_filter(
        filter_expression, expression_names, expression_values, input_type
    )

    scan_kwargs = {
        "FilterExpression": filter_expression,
        "ExpressionAttributeNames": expression_names,
        "ExpressionAttributeValues": expression_values,
    }
    items = scan_all_items(scan_kwargs)
    if not items:
        return {}
    latest_item = max(items, key=lambda x: x["timestamp"])
    return latest_item

def get_metrics_period(metricType, timeRange, streaming, input_type):
    time_ranges = {
        "week": timedelta(weeks=1),
        "month": timedelta(days=30),
        "three-month": timedelta(days=90),
        "max": None
    }

    if timeRange not in time_ranges:
        return {"error": f"Invalid timeRange. Valid options: {list(time_ranges.keys())}"}

    end_date = datetime.now()
    start_date = end_date - time_ranges[timeRange] if timeRange != "max" else datetime(2025, 3, 10)
    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    filter_expression = "#ts BETWEEN :start_date AND :end_date AND #streaming = :streaming AND #model_key = :common"
    expression_names = {
        "#ts": "timestamp",
        "#streaming": "streaming",
        "#model_key": "model_key"
    }
    expression_values = {
        ":start_date": start_date_str,
        ":end_date": end_date_str,
        ":streaming": streaming,
        ":common": "common"
    }
    filter_expression, expression_names, expression_values = add_input_type_filter(
        filter_expression, expression_names, expression_values, input_type
    )
    scan_kwargs = {
        "FilterExpression": filter_expression,
        "ExpressionAttributeNames": expression_names,
        "ExpressionAttributeValues": expression_values,
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


def get_metrics_by_date(metricType, date, streaming, input_type):
    try:
        start_date = datetime.strptime(date, "%d-%m-%Y")
        end_date = start_date + timedelta(days=1)
    except ValueError:
        return {"error": "Invalid date format. Use '12-12-2024'."}

    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    filter_expression = "#ts BETWEEN :start_date AND :end_date AND #streaming = :streaming AND #model_key = :common"
    expression_names = {
        "#ts": "timestamp",
        "#streaming": "streaming",
        "#model_key": "model_key"
    }
    expression_values = {
        ":start_date": start_date_str,
        ":end_date": end_date_str,
        ":streaming": streaming,
        ":common": "common"
    }
    filter_expression, expression_names, expression_values = add_input_type_filter(
        filter_expression, expression_names, expression_values, input_type
    )
    scan_kwargs = {
        "FilterExpression": filter_expression,
        "ExpressionAttributeNames": expression_names,
        "ExpressionAttributeValues": expression_values,
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
    vllm_item = get_latest_vllm(streaming, input_type)
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
        input_type = params.get("inputType", "static").lower() == "static" 

        if not metricType or not timeRange:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing metricType or timeRange parameter"})}

        response = get_metrics_period(metricType, timeRange, streaming, input_type)
        return {"statusCode": 200, "body": json.dumps(response)}

    elif path == "/default/metrics/date":
        metricType = params.get("metricType")
        date = params.get("date")
        streaming = params.get("streaming", "true").lower() == "true"
        input_type = params.get("inputType", "static").lower() == "static"

        if not metricType or not date:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing metricType or date parameter"})}

        response = get_metrics_by_date(metricType, date, streaming, input_type)
        return {"statusCode": 200, "body": json.dumps(response)}
    elif path == "/default/metrics/vllm":
        streaming = params.get("streaming", "true").lower() == "true"
        input_type = params.get("inputType", "static").lower() == "static"
        vllm_item = get_latest_vllm(streaming, input_type)
        print(vllm_item)
        if not vllm_item:
            return {"statusCode": 404, "body": json.dumps({"error": "No vLLM metrics found"})}
        return {"statusCode": 200, "body": json.dumps(vllm_item)}

    else:
        return {"statusCode": 404, "body": json.dumps({"error": f"Invalid route: {path}"})}
