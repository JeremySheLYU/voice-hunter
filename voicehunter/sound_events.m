// Native event classifier. SoundAnalysis returns time labels, not separated audio.
#import <Foundation/Foundation.h>
#import <SoundAnalysis/SoundAnalysis.h>
#import <CoreMedia/CoreMedia.h>

@interface EventObserver : NSObject<SNResultsObserving>
@property(nonatomic, strong) NSMutableArray *windows;
@property(nonatomic, strong) NSError *failure;
@property(nonatomic, strong) NSSet<NSString *> *targets;
@end

@implementation EventObserver
- (void)request:(id<SNRequest>)request didProduceResult:(id<SNResult>)result {
    if (![result isKindOfClass:SNClassificationResult.class]) return;
    SNClassificationResult *r = (SNClassificationResult *)result;
    double score = 0;
    NSMutableDictionary *classScores = [NSMutableDictionary dictionary];
    for (SNClassification *c in r.classifications) {
        if ([self.targets containsObject:c.identifier]) {
            classScores[c.identifier] = @(c.confidence);
            score = MAX(score, c.confidence);
        }
    }
    [self.windows addObject:@{
        @"start": @(CMTimeGetSeconds(r.timeRange.start)),
        @"duration": @(CMTimeGetSeconds(r.timeRange.duration)), @"score": @(score), @"class_scores": classScores
    }];
}
- (void)request:(id<SNRequest>)request didFailWithError:(NSError *)error { self.failure = error; }
- (void)requestDidComplete:(id<SNRequest>)request {}
@end

static int fail(NSString *message) {
    fprintf(stderr, "%s\n", message.UTF8String);
    return 1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) return fail(@"Expected --labels, or an audio path and one or more known sound labels.");
        NSError *err = nil;
        SNClassifySoundRequest *request = [[SNClassifySoundRequest alloc]
            initWithClassifierIdentifier:SNClassifierIdentifierVersion1 error:&err];
        if (!request) return fail(err.localizedDescription);
        id payload = nil;
        if (!strcmp(argv[1], "--labels")) {
            payload = request.knownClassifications;
        } else {
            if (argc < 3) return fail(@"Expected at least one target label.");
            NSString *path = [NSString stringWithUTF8String:argv[1]];
            SNAudioFileAnalyzer *analyzer = [[SNAudioFileAnalyzer alloc] initWithURL:
                [NSURL fileURLWithPath:path] error:&err];
            if (!analyzer) return fail(err.localizedDescription);
            EventObserver *observer = [EventObserver new];
            observer.windows = [NSMutableArray array];
            NSMutableArray *labels = [NSMutableArray array];
            for (int i=2; i<argc; i++) {
                NSString *label = [NSString stringWithUTF8String:argv[i]];
                if ([request.knownClassifications containsObject:label]) [labels addObject:label];
            }
            if (!labels.count) return fail(@"No requested labels are supported by this classifier.");
            observer.targets = [NSSet setWithArray:labels];
            request.windowDuration = CMTimeMakeWithSeconds(0.975, 48000);
            request.overlapFactor = 0.5;
            if (![analyzer addRequest:request withObserver:observer error:&err]) return fail(err.localizedDescription);
            [analyzer analyze];
            if (observer.failure) return fail(observer.failure.localizedDescription);
            payload = @{@"windows": observer.windows, @"labels":labels};
        }
        NSData *json = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&err];
        if (!json) return fail(err.localizedDescription);
        fwrite(json.bytes, 1, json.length, stdout);
        return 0;
    }
}
