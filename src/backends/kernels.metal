#include <metal_stdlib>
using namespace metal;
struct Params { uint n,rows,cols,batch,heads,kv_heads,dim,tokens,position,operation; float epsilon,theta; };
kernel void matmul(device const float* w [[buffer(0)]],device const float* x [[buffer(1)]],device float* y [[buffer(2)]],constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.batch)return;uint r=i%p.rows,b=i/p.rows;float sum=0;
    for(uint c=0;c<p.cols;++c)sum+=w[r*p.cols+c]*x[b*p.cols+c];y[i]=sum;
}
kernel void elementwise(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device float* y [[buffer(2)]],constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.n)return;
    if(p.operation==0)y[i]=a[i]+b[i];else if(p.operation==1)y[i]=a[i]*b[i];else y[i]=a[i]/(1+exp(-a[i]));
}
kernel void rmsnorm(device const float* x [[buffer(0)]],device const float* w [[buffer(1)]],device float* y [[buffer(2)]],constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.n)return;float sum=0;for(uint j=0;j<p.n;++j)sum+=x[j]*x[j];y[i]=x[i]*rsqrt(sum/float(p.n)+p.epsilon)*w[i];
}
kernel void rope(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],constant Params& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.heads*p.dim/2)return;uint h=i/(p.dim/2),d=i%(p.dim/2),offset=h*p.dim+d;
    float angle=float(p.position)*pow(p.theta,-2.f*float(d)/float(p.dim)),c=cos(angle),s=sin(angle),a=x[offset],b=x[offset+p.dim/2];
    y[offset]=a*c-b*s;y[offset+p.dim/2]=b*c+a*s;
}
kernel void softmax(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],constant Params& p [[buffer(2)]],uint row [[thread_position_in_grid]]) {
    if(row>=p.rows)return;uint offset=row*p.n;float maximum=-INFINITY;
    for(uint i=0;i<p.n;++i)maximum=max(maximum,x[offset+i]);float sum=0;
    for(uint i=0;i<p.n;++i){y[offset+i]=exp(x[offset+i]-maximum);sum+=y[offset+i];}
    for(uint i=0;i<p.n;++i)y[offset+i]/=sum;
}
kernel void attention_scores(device const float* q [[buffer(0)]],device const float* k [[buffer(1)]],device float* scores [[buffer(2)]],constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.heads*p.tokens)return;uint h=i/p.tokens,t=i%p.tokens,kh=h/(p.heads/p.kv_heads);float sum=0;
    for(uint d=0;d<p.dim;++d)sum+=q[h*p.dim+d]*k[(t*p.kv_heads+kh)*p.dim+d];scores[i]=sum/sqrt(float(p.dim));
}
kernel void attention_values(device const float* probs [[buffer(0)]],device const float* v [[buffer(1)]],device float* y [[buffer(2)]],constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.heads*p.dim)return;uint h=i/p.dim,d=i%p.dim,kh=h/(p.heads/p.kv_heads);float sum=0;
    for(uint t=0;t<p.tokens;++t)sum+=probs[h*p.tokens+t]*v[(t*p.kv_heads+kh)*p.dim+d];y[i]=sum;
}
