import AppMetricsPage from "../sections/@dashboard/app/AppMetricsPage";

export default function ResponseTimes() {
  return (
    <div>
      <div>
        <p>AWS_ACCESS_KEY_ID: {process.env.REACT_APP_AWS_ACCESS_KEY_ID}</p>
        <p>AWS_SECRET_ACCESS_KEY: {process.env.REACT_APP_AWS_SECRET_ACCESS_KEY}</p>
        <p>AWS_REGION: {process.env.REACT_APP_AWS_REGION}</p>
        <p>AWS_BEDROCK_ACCESS_KEY_ID: {process.env.REACT_APP_AWS_BEDROCK_ACCESS_KEY_ID}</p>
        <p>AWS_BEDROCK_SECRET_ACCESS_KEY: {process.env.REACT_APP_AWS_BEDROCK_SECRET_ACCESS_KEY}</p>
        <p>AWS_BEDROCK_REGION: {process.env.REACT_APP_AWS_BEDROCK_REGION}</p>
      </div>
      
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="response_times" title="Response Times Metrics" metricName="Response Times" min={100} cdf />
      </div>
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="response_times_median" title="Response Times Metrics Median Metrics" metricName="Response Times Metrics Median" />
      </div>
      <div>
        <AppMetricsPage metricType="response_times_p95" title="Response Times Metrics P95 Metrics" metricName="Response Times Metrics P95" />
      </div>
    </div>
  );
}
