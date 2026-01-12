{{/*
Expand the name of the chart.
*/}}
{{- define "sherman-qc.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "sherman-qc.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "sherman-qc.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sherman-qc.labels" -}}
helm.sh/chart: {{ include "sherman-qc.chart" . }}
{{ include "sherman-qc.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sherman-qc.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sherman-qc.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Backend labels
*/}}
{{- define "sherman-qc.backend.labels" -}}
{{ include "sherman-qc.labels" . }}
app.kubernetes.io/component: backend
{{- end }}

{{/*
Backend selector labels
*/}}
{{- define "sherman-qc.backend.selectorLabels" -}}
{{ include "sherman-qc.selectorLabels" . }}
app.kubernetes.io/component: backend
{{- end }}

{{/*
Frontend labels
*/}}
{{- define "sherman-qc.frontend.labels" -}}
{{ include "sherman-qc.labels" . }}
app.kubernetes.io/component: frontend
{{- end }}

{{/*
Frontend selector labels
*/}}
{{- define "sherman-qc.frontend.selectorLabels" -}}
{{ include "sherman-qc.selectorLabels" . }}
app.kubernetes.io/component: frontend
{{- end }}

{{/*
Worker labels
*/}}
{{- define "sherman-qc.worker.labels" -}}
{{ include "sherman-qc.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "sherman-qc.worker.selectorLabels" -}}
{{ include "sherman-qc.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "sherman-qc.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "sherman-qc.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Database URL - validates that exactly one database option is enabled
*/}}
{{- define "sherman-qc.databaseUrl" -}}
{{- if and (not .Values.postgresql.enabled) (not .Values.externalPostgresql.enabled) }}
{{- fail "Either postgresql.enabled or externalPostgresql.enabled must be true" }}
{{- end }}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:$(DATABASE_PASSWORD)@{{ include "sherman-qc.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else }}
postgresql://{{ .Values.externalPostgresql.username }}:$(DATABASE_PASSWORD)@{{ .Values.externalPostgresql.host }}:{{ .Values.externalPostgresql.port }}/{{ .Values.externalPostgresql.database }}
{{- end }}
{{- end }}

{{/*
Redis URL - validates that Redis is available when workers are enabled
*/}}
{{- define "sherman-qc.redisUrl" -}}
{{- if .Values.redis.enabled }}
redis://:$(REDIS_PASSWORD)@{{ include "sherman-qc.fullname" . }}-redis-master:6379
{{- else if .Values.externalRedis.enabled }}
redis://:$(REDIS_PASSWORD)@{{ .Values.externalRedis.host }}:{{ .Values.externalRedis.port }}
{{- end }}
{{- end }}

{{/*
Validate worker dependencies
*/}}
{{- define "sherman-qc.validateWorkerDeps" -}}
{{- if and .Values.worker.enabled (not (or .Values.redis.enabled .Values.externalRedis.enabled)) }}
{{- fail "worker.enabled requires redis.enabled or externalRedis.enabled to be true" }}
{{- end }}
{{- end }}
