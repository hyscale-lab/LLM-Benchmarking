import AppMetricsPage from "../sections/@dashboard/app/AppMetricsPage";

export default function TimeBetweenToken() {
  return (
    <div>
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="timebetweentokens" title="Time Between Tokens Metrics" metricName="Time Between Tokens" min={0} cdf />
      </div>
      <div style={{ paddingBottom: "30px" }}>
        <AppMetricsPage metricType="timebetweentokens_median" title="Time Between Tokens Median Metrics" metricName="Time Between Tokens Median" />
      </div>
      <div>
        <AppMetricsPage metricType="timebetweentokens_p95" title="Time Between Tokens P95 Metrics" metricName="Time Between Tokens P95" />
      </div>
    </div>
  );
}
